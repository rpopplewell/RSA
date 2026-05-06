//! PKCS#1 v1.5 support as described in [RFC8017 § 8.2].
//!
//! # Usage
//!
//! See [code example in the toplevel rustdoc](../index.html#pkcs1-v15-signatures).
//!
//! [RFC8017 § 8.2]: https://datatracker.ietf.org/doc/html/rfc8017#section-8.2

use alloc::vec::Vec;
use const_oid::AssociatedOid;
use core::sync::atomic::compiler_fence;
use core::sync::atomic::Ordering::SeqCst;
use crypto_bigint::{BoxedUint, Choice, CtAssign, CtEq, CtLt, CtSelect};
use digest::{Digest, KeyInit};
use hmac::{Hmac, Mac};
use rand_core::TryCryptoRng;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::algorithms::pad::uint_to_zeroizing_be_pad;
use crate::errors::{Error, Result};
use crate::traits::keys::PrivateKeyParts;

/// Implicit Rejection Pseudo-Random Function (IRPRF)
///
/// Fills `out` using HMAC-SHA256
///
/// Returns an error if `kdk` is not 32 bytes or `out` exceeds 8192 bytes (k > 65536 bits);
/// both checks are required by the specification.
///
/// See https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-08.html#name-implicit-rejection-pseudo-r
///
fn irprf(kdk: &[u8], label: &[u8], out: &mut [u8]) -> Result<()> {
    if kdk.len() != 32 || out.len() > 8192 {
        return Err(Error::Decryption);
    }

    let bit_length = ((out.len() * 8) as u16).to_be_bytes();
    let mac_init = Hmac::<Sha256>::new_from_slice(kdk).unwrap();

    for (i, chunk) in out.chunks_mut(32).enumerate() {
        let block = mac_init
            .clone()
            .chain_update((i as u16).to_be_bytes())
            .chain_update(label)
            .chain_update(bit_length)
            .finalize()
            .into_bytes();
        chunk.copy_from_slice(&block[..chunk.len()]);
    }

    Ok(())
}

/// Derives the alternative message and length for PKCS#1 v1.5 implicit rejection.
///
/// Fills `am` with the k-byte alternative message and returns the alternative length AL.
/// `am[k - AL..]` is the fallback plaintext when padding is invalid.
///
/// Returns an error if the keysize k > 65536 bits.
///
/// See https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-08.html#section-7.2-3.3.1
///
fn derive_am(
    am: &mut [u8],
    priv_key: &impl PrivateKeyParts,
    ciphertext: &BoxedUint,
) -> Result<usize> {
    const LENGTH_LABEL: &[u8] = b"length";
    const MESSAGE_LABEL: &[u8] = b"message";
    let k: usize = priv_key.size();

    // Step 1a: D = I2OSP(d, k)
    let d = Zeroizing::new(uint_to_zeroizing_be_pad(priv_key.d().clone(), k)?);

    // Step 1b: DH = SHA256(D)
    let dh = Zeroizing::new(<[u8; 32]>::from(Sha256::digest(&d)));

    // Step 1c: KDK = HMAC(DH, C, SHA256)
    let c_bytes = ciphertext.to_be_bytes();
    let kdk = Zeroizing::new(<[u8; 32]>::from(
        Hmac::<Sha256>::new_from_slice(dh.as_ref())
            .unwrap()
            .chain_update(&c_bytes[c_bytes.len() - k..])
            .finalize()
            .into_bytes(),
    ));

    // Step 2a: CL = IRPRF(KDK, "length", 256)
    let mut cl = Zeroizing::new([0u8; 256]);
    irprf(kdk.as_ref(), LENGTH_LABEL, cl.as_mut())?;

    // Step 2b: AM = IRPRF(KDK, "message", k)
    irprf(kdk.as_ref(), MESSAGE_LABEL, am)?;

    // Step 3: select the last candidate <= max_len.
    // The mask pre-screens candidates by zeroing bits above max_len's highest set bit,
    // avoiding an obvious-overflow branch before the ct_lt comparison.
    let mut al = 0u16;
    let max_len = (k - 11) as u16;
    let mask = u16::MAX >> max_len.leading_zeros();

    for chunk in cl.chunks_exact(2) {
        let candidate = u16::from_be_bytes([chunk[0], chunk[1]]) & mask;
        let is_valid = !u16::ct_lt(&max_len, &candidate);

        al = u16::ct_select(&al, &candidate, is_valid);
    }

    Ok(al as usize)
}

/// Fills the provided slice with random values, which are guaranteed
/// to not be zero.
#[inline]
fn non_zero_random_bytes<R: TryCryptoRng + ?Sized>(
    rng: &mut R,
    data: &mut [u8],
) -> core::result::Result<(), R::Error> {
    rng.try_fill_bytes(data)?;

    for el in data {
        if *el == 0u8 {
            // TODO: break after a certain amount of time
            while *el == 0u8 {
                rng.try_fill_bytes(core::slice::from_mut(el))?;
            }
        }
    }

    Ok(())
}

/// Applied the padding scheme from PKCS#1 v1.5 for encryption.  The message must be no longer than
/// the length of the public modulus minus 11 bytes.
pub(crate) fn pkcs1v15_encrypt_pad<R>(
    rng: &mut R,
    msg: &[u8],
    k: usize,
) -> Result<Zeroizing<Vec<u8>>>
where
    R: TryCryptoRng + ?Sized,
{
    if msg.len() + 11 > k {
        return Err(Error::MessageTooLong);
    }

    // EM = 0x00 || 0x02 || PS || 0x00 || M
    let mut em = Zeroizing::new(vec![0u8; k]);
    em[1] = 2;
    non_zero_random_bytes(rng, &mut em[2..k - msg.len() - 1]).map_err(|_: R::Error| Error::Rng)?;
    em[k - msg.len() - 1] = 0;
    em[k - msg.len()..].copy_from_slice(msg);
    Ok(em)
}

/// Removes the encryption padding scheme from PKCS#1 v1.5
///
/// Returns the plaintext if the padding is valid, and an alternative plaintext
/// if the padding is invalid.
///
/// See https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-08.html#section-7.2-3.4.1
///
/// Note that whether this function returns an error or not discloses secret
/// information. If an attacker can cause this function to run repeatedly and
/// learn whether each instance returned an error then they can decrypt and
/// forge signatures as if they had the private key. See
/// `decrypt_session_key` for a way of solving this problem.
#[inline]
pub(crate) fn pkcs1v15_implicit_rejection(
    em: &[u8],
    priv_key: &impl PrivateKeyParts,
    ciphertext: &BoxedUint,
) -> Result<Vec<u8>> {
    let k = priv_key.size();

    let mut am = Zeroizing::new(vec![0u8; k]);
    let al = derive_am(&mut am, priv_key, ciphertext)?;

    let (valid, l) = decrypt_inner(em, k)?;
    let msg_len = usize::from(u16::ct_select(&(al as u16), &(l as u16), valid));

    // Touch one byte per cache line (conservatively 32 bytes) in am and em to warm the cache independent of msg_len.
    let mut _sink = 0u8;
    for i in (0..k).step_by(32) {
        unsafe {
            _sink ^= core::ptr::read_volatile(&am[i]) ^ core::ptr::read_volatile(&em[i]);
        }
    }

    compiler_fence(SeqCst);

    // First pick between em/am (valid/invalid path), then between that byte and 0 (in-message vs out-of-message).
    // The clamped src index when j >= msg_len is discarded by ct_select, and .min() compiles to a cmov.
    //
    // We iterate over the entire max_len to pass class 5 probes of the marvin-toolkit, which uses msg_len = 0.
    let max_len = k - 11;
    let mut result = vec![0u8; max_len];
    for j in 0..max_len {
        let in_msg = u16::ct_lt(&(j as u16), &(msg_len as u16));
        let src = (k - msg_len + j).min(k - 1);
        let selected = u8::ct_select(&am[src], &em[src], valid);
        result[j] = u8::ct_select(&0u8, &selected, in_msg);
    }
    result.truncate(msg_len);
    Ok(result)
}

/// Removes the PKCS1v15 padding It returns one or zero in valid that indicates whether the
/// plaintext was correctly structured. In either case, the plaintext is
/// returned in em so that it may be read independently of whether it was valid
/// in order to maintain constant memory access patterns. If the plaintext was
/// valid then index contains the index of the original message in em.
///
/// See https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-08.html#section-7.2-3.4.1
///
#[inline]
fn decrypt_inner(em: &[u8], k: usize) -> Result<(Choice, usize)> {
    if k < 11 {
        return Err(Error::Decryption);
    }

    let first_byte_is_zero = em[0].ct_eq(&0u8);
    let second_byte_is_two = em[1].ct_eq(&2u8);

    // The remainder of the plaintext must be a string of non-zero random
    // octets, followed by a 0, followed by the message.
    //   looking_for_index: 1 iff we are still looking for the zero.
    //   index: the offset of the first zero byte.
    let mut looking_for_index = Choice::TRUE;
    let mut index = 0u32;

    for (i, el) in em.iter().enumerate().skip(2) {
        let equals0 = el.ct_eq(&0u8);
        index.ct_assign(&(i as u32), looking_for_index & equals0);
        looking_for_index &= !equals0;
    }

    // EM = 0x00 || 0x02 || PS (>=8 non-zero bytes) || 0x00 || M
    // PS must be at least 8 bytes; it starts at byte 2, so the separator must be at index >= 10.
    let valid_ps = !u32::ct_lt(&index, &10);
    let valid = first_byte_is_zero & second_byte_is_two & !looking_for_index & valid_ps;
    index = u32::ct_select(&0, &(index + 1), valid);

    let l = k - index as usize;

    Ok((valid, l))
}

#[inline]
pub(crate) fn pkcs1v15_sign_pad(prefix: &[u8], hashed: &[u8], k: usize) -> Result<Vec<u8>> {
    let hash_len = hashed.len();
    let t_len = prefix.len() + hashed.len();
    if k < t_len + 11 {
        return Err(Error::MessageTooLong);
    }

    // EM = 0x00 || 0x01 || PS || 0x00 || T
    let mut em = vec![0xff; k];
    em[0] = 0;
    em[1] = 1;
    em[k - t_len - 1] = 0;
    em[k - t_len..k - hash_len].copy_from_slice(prefix);
    em[k - hash_len..k].copy_from_slice(hashed);

    Ok(em)
}

#[inline]
pub(crate) fn pkcs1v15_sign_unpad(prefix: &[u8], hashed: &[u8], em: &[u8], k: usize) -> Result<()> {
    let hash_len = hashed.len();
    let t_len = prefix.len() + hashed.len();
    if k < t_len + 11 {
        return Err(Error::Verification);
    }

    // EM = 0x00 || 0x01 || PS || 0x00 || T
    let mut ok = em[0].ct_eq(&0u8);
    ok &= em[1].ct_eq(&1u8);
    ok &= em[k - hash_len..k].ct_eq(hashed);
    ok &= em[k - t_len..k - hash_len].ct_eq(prefix);
    ok &= em[k - t_len - 1].ct_eq(&0u8);

    for el in em.iter().skip(2).take(k - t_len - 3) {
        ok &= el.ct_eq(&0xff)
    }

    // TODO(tarcieri): avoid branching here by e.g. using a pseudorandom rejection symbol
    if !ok.to_bool() {
        return Err(Error::Verification);
    }

    Ok(())
}

/// prefix = 0x30 <oid_len + 8 + digest_len> 0x30 <oid_len + 4> 0x06 <oid_len> oid 0x05 0x00 0x04 <digest_len>
#[inline]
pub(crate) fn pkcs1v15_generate_prefix<D>() -> Vec<u8>
where
    D: Digest + AssociatedOid,
{
    let oid = D::OID.as_bytes();
    let oid_len = oid.len() as u8;
    let digest_len = <D as Digest>::output_size() as u8;
    let mut v = vec![
        0x30,
        oid_len + 8 + digest_len,
        0x30,
        oid_len + 4,
        0x6,
        oid_len,
    ];
    v.extend_from_slice(oid);
    v.extend_from_slice(&[0x05, 0x00, 0x04, digest_len]);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn test_non_zero_bytes() {
        for _ in 0..10 {
            let mut rng = ChaCha8Rng::from_seed([42; 32]);
            let mut b = vec![0u8; 512];
            non_zero_random_bytes(&mut rng, &mut b).unwrap();
            for el in &b {
                assert_ne!(*el, 0u8);
            }
        }
    }

    #[test]
    fn test_encrypt_tiny_no_crash() {
        let mut rng = ChaCha8Rng::from_seed([42; 32]);
        let k = 8;
        let message = vec![1u8; 4];
        let res = pkcs1v15_encrypt_pad(&mut rng, &message, k);
        assert_eq!(res, Err(Error::MessageTooLong));
    }
}

use crate::dftd3::reference3_inc::{C6AB};

pub const MREF: usize = 5;

pub fn get_c6(iref: usize, jref: usize, ati: u8, atj: u8) -> f64 {
    // Index arithmetic in usize: `z * (z - 1)` overflows u8 for Z >= 17
    // (Cl: 17*16 = 272 > 255). Widen before multiplying.
    let (zi, zj) = (ati as usize, atj as usize);
    if zi > zj {
        let ic = zj + zi * (zi - 1) / 2 - 1;
        C6AB[ic][jref][iref]
    } else {
        let ic = zi + zj * (zj - 1) / 2 - 1;
        C6AB[ic][iref][jref]
    }
}


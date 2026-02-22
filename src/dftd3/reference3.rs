use crate::dftd3::reference3_inc::{C6AB};

pub const MREF: usize = 5;

pub fn get_c6(iref: usize, jref: usize, ati: u8, atj: u8) -> f64 {
    let (ic, c6): (u8, f64);

    if ati > atj {
        ic = atj + ati * (ati - 1)/2 - 1;
        c6 = C6AB[ic as usize][jref][iref];
    } else {
        ic = ati + atj * (atj - 1)/2 - 1;
        c6 = C6AB[ic as usize][iref][jref];
    };

   c6
}


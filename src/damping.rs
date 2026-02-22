// /// Logic exercise to distribute a triple energy to atomwise energies.
// pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
//     // atom indices: ii, jj, kk
//     let triple = match ii == jj {
//         true => {
//             match ii == kk {
//                 true => 1.0/6.0,
//                 false => 0.5,
//             }
//         },
//         false => {
//             match ii != kk && jj != kk {
//                 true => 1.0,
//                 false => 0.5,
//             }
//         }
//     };
//     triple
// }

/// Logic exercise to distribute a triple energy to atomwise energies.
#[inline]
pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
    // atom indices: ii, jj, kk
    let equals: u8 = ((ii == jj) as u8) + ((ii == kk) as u8) + ((jj == kk) as u8);
    let triple = match equals {
        0 => 1.0,
        1 => 0.5,
        _ => 1.0/6.0,
    };
    triple
}

#[cfg(test)]
mod tests {
    use crate::damping::triple_scale;
    use approx::AbsDiffEq;

    #[test]
    fn triple_scale_num() -> () {
        let epsilon = 1.0e-15;
        assert!(triple_scale(2, 2, 2).abs_diff_eq(&(1.0/6.0), epsilon));
        assert!(triple_scale(3, 3, 4).abs_diff_eq(&0.5, epsilon));
        assert!(triple_scale(5, 6, 7).abs_diff_eq(&1.0, epsilon));
        assert!(triple_scale(1, 8, 8).abs_diff_eq(&0.5, epsilon));
        assert!(triple_scale(9, 7, 9).abs_diff_eq(&0.5, epsilon));
    }
}
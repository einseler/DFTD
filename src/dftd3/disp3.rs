use ndarray::{Array, Array1, Array2, Array3};
use crate::model::Molecule;
use crate::dftd3::model3::D3Model;
use crate::dftd3::damping3::RationalDamping3Param;
use crate::cutoff::{RealspaceCutoff, get_lattice_points_cutoff};
use crate::dftd3::ncoord3::{get_coordination_number3, get_coordination_number3_grad};
use nalgebra::Matrix3xX;
use crate::disp::DispersionResult;

/// Wrapper to handle the evaluation of dispersion energy and derivatives.
pub fn get_dispersion(
    mol: &mut Molecule,
    disp: &D3Model,
    param: &RationalDamping3Param,
    cutoff: &RealspaceCutoff,
    atm: bool,
    grad: bool,
) -> DispersionResult {
    if grad {
        // Obtain the lattice with the cutoff radius for calculating coordination numbers
        // and calculate coordination numbers as well as their derivatives w.r.t. the Cartesian coordinates
        // and strain deformations. The CNs are saved in atoms and the derivatives in arrays.
        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        let (dcndr, dcndl): (Array3<f64>, Array3<f64>) = get_coordination_number3_grad(mol, &lattr, cutoff.cn, disp.rcov.view());

        // Calculate weight references and their derivatives w.r.t. CN. These are saved them in atoms.
        disp.weight_references_grad(mol);

        // Calculate C6 parameters and their derivatives w.r.t. CN.
        let (c6, dc6dcn): (Array2<f64>, Array2<f64>) = disp.get_atomic_c6_grad(&mol);

        // Initialise the energy vector, its derivative w.r.t. CN, the gradient
        // and sigma matrix as a zero-matrices.
        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);       // E(CN(r, L), r, L)
        let mut dedcn: Array1<f64> = Array::zeros(mol.n_atoms);          // ∂E/∂(CN)
        let mut gradient: Array2<f64> = Array::zeros((mol.n_atoms, 3));  // dE/dr
        let mut sigma: Array2<f64> = Array::zeros((3, 3));               // dE/dL

        // Obtain the lattice with the cutoff radius for calculating 2-body dispersion
        // and evaluate 2-body dispersion.
        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp2);
        param.get_dispersion2_grad_par(mol, &lattr, cutoff.disp2, c6.view(), dc6dcn.view(),
                                   energies.view_mut(), dedcn.view_mut(), gradient.view_mut(), sigma.view_mut());

        if atm {
            // Obtain the lattice with the cutoff radius for calculating 3-body dispersion
            // and evaluate Axilrod-Teller-Muto (ATM) 3-body dispersion.
            let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp3);
            param.get_dispersion3_grad(mol, &lattr, cutoff.disp2, c6.view(), dc6dcn.view(),
                                       energies.view_mut(), dedcn.view_mut(), gradient.view_mut(), sigma.view_mut());
        }

        // The gradient: dE/dr = (∂E/∂(CN)) (d(CN)/dr) + ∂E/∂r.
        // So far, only ∂E/∂r is calculated. The first term has to be added to it.
        // call d3_gemv(dcndr, dEdcn, gradient, beta=1.0_wp)
        gradient += &(dedcn.dot(&dcndr.into_shape((mol.n_atoms, mol.n_atoms*3)).unwrap()).into_shape((mol.n_atoms, 3)).unwrap());

        // The sigma matrix: dE/dL = (∂E/∂(CN)) (d(CN)/dL) + ∂E/∂L.
        // So far, only ∂E/∂L is calculated. The first term has to be added to it.
        // call d3_gemv(dcndL, dEdcn, sigma, beta=1.0_wp)
        sigma += &(dedcn.dot(&dcndl.into_shape((mol.n_atoms, 3*3)).unwrap()).into_shape((3, 3)).unwrap());

        DispersionResult{
            energy: energies.sum(),
            gradient: Some(gradient),
            sigma: Some(sigma),
        }
    } else {
        // Obtain the lattice with the cutoff radius for calculating coordination numbers
        // and calculate coordination numbers. These are saved in atoms.
        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        get_coordination_number3(mol, &lattr, cutoff.cn, disp.rcov.view());

        // Calculate weight references and save them in atoms.
        disp.weight_references(mol);

        // Calculate C6 parameters.
        let c6: Array2<f64> = disp.get_atomic_c6(&mol);

        // Initialise the energy vector as a zero-vector.
        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);

        // Obtain the lattice with the cutoff radius for calculating 2-body dispersion
        // and evaluate 2-body dispersion.
        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp2);
        param.get_dispersion2_par(mol, &lattr, cutoff.disp2, c6.view(), energies.view_mut());

        if atm {
            // Obtain the lattice with the cutoff radius for calculating 3-body dispersion
            // and evaluate Axilrod-Teller-Muto (ATM) 3-body dispersion.
            let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp3);
            param.get_dispersion3(mol, &lattr, cutoff.disp3, c6.view(), energies.view_mut());
        }

        DispersionResult{
            energy: energies.sum(),
            gradient: None,
            sigma: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test::get_uracil;
    use crate::model::Molecule;
    use crate::dftd3::model3::D3Model;
    use crate::dftd3::damping3::{D3Param, RationalDamping3Param};
    use crate::cutoff::{RealspaceCutoff, RealspaceCutoffBuilder};
    use crate::dftd3::disp3::get_dispersion;
    use approx::AbsDiffEq;
    use ndarray::{array, Array, Array2};

    #[test]
    fn d3_dispersion() -> () {
        let mut mol: Molecule = get_uracil();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3Param{
            s6: 1.0,
            s8: 1.2177,
            s9: 1.0,
            rs6: 1.0,
            rs8: 1.0,
            a1: 0.4145,
            a2: 4.8593,
            alp: 14.0,
            bet: 0.0,
        };
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();

        let disp_result = get_dispersion(&mut mol, &disp, &param, &cutoff, true, false);
        let energy = disp_result.energy;

        let energy_ref =  -0.0273470728456744;

        assert!(energy.abs_diff_eq(&energy_ref, 1e-15));
    }

    #[test]
    fn d3_dispersion_derivative() -> () {
        let mut mol: Molecule = get_uracil();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3Param{
            s6: 1.0,
            s8: 1.2177,
            s9: 1.0,
            rs6: 1.0,
            rs8: 1.0,
            a1: 0.4145,
            a2: 4.8593,
            alp: 14.0,
            bet: 0.0,
        };
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();

        let disp_result = get_dispersion(&mut mol, &disp, &param, &cutoff, true, true);
        let energy = disp_result.energy;
        let gradient = disp_result.gradient.unwrap();
        let sigma = disp_result.sigma.unwrap();

        let energy_ref =  -0.0273470728456744;

        let gradient_ref_array: [[f64; 24]; 3] = [
            [ -0.0002922257091589, -0.0004397054551162, -0.0002422124227591, -0.0000396519088368,
                -0.0001037958645466, -0.0001185328848082, -0.0003055817046096,  0.0002244898884948,
                0.0000481376843799,  0.0000018192488779, -0.0001944402854162, -0.0001042233039987,
                -0.0000544319143352,  0.0000538870833464,  0.0001563558672101,  0.0002271448388504,
                0.0000475688958963, -0.0001013385899268,  0.0003622301337491,  0.0000818794070812,
                0.0002517524644230,  0.0002233673849937,  0.0000006540887190,  0.0003168530574906,],
            [  0.0002860750867337,  0.0001508802900668,  0.0000740238840314,  0.0000541534239020,
                0.0002518859746938,  0.0003873830116156,  0.0002490116688516,  0.0002388145809157,
                -0.0000338006636899,  0.0002275096740808, -0.0000094952933170, -0.0000948793833258,
                0.0000092093455622, -0.0000862042309876,  0.0001648674054662, -0.0001191254390757,
                -0.0003057751119419, -0.0001281207963425, -0.0000959966354563, -0.0004000078861741,
                -0.0003135007649217,  0.0000084544826712, -0.0002179004078622, -0.0002974622154962,],
            [ -0.0003084328540801, -0.0002921881545305, -0.0004395619530745, -0.0004572888483781,
                -0.0004235131706108, -0.0002364887276760,  0.0000016391535923, -0.0002767855521256,
                -0.0002130626034150, -0.0000435428832491, -0.0000594158548938, -0.0001579660936698,
                0.0001943627039027,  0.0005279760560281,  0.0003243407675177,  0.0004440770599671,
                0.0003852573582758,  0.0001271983381940,  0.0002916059588154,  0.0002206651535565,
                0.0003251335164345,  0.0000705760668356,  0.0000278342649994, -0.0000324197024157,],
        ];
        let mut gradient_ref: Array2<f64> = Array::zeros((3, 24));
        for i in 0..3 {
            for j in 0..24 {
                gradient_ref[[i, j]] = gradient_ref_array[i][j];
            }
        }
        let gradient_ref: Array2<f64> = gradient_ref.reversed_axes();

        let sigma_ref: Array2<f64> = array![
            [  0.0119407601200530, -0.0032620922167762,  0.0046766272572748,],
            [ -0.0032620922167762,  0.0126508050097128, -0.0054248630102770,],
            [  0.0046766272572748, -0.0054248630102770,  0.0186926773035078,],
        ].reversed_axes();

        assert!(energy.abs_diff_eq(&energy_ref, 1e-15));
        assert!(gradient.abs_diff_eq(&gradient_ref, 1e-15));
        assert!(sigma.abs_diff_eq(&sigma_ref, 1e-15));
    }
}
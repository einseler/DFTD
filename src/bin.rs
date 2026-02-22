use chemfiles::Frame;
use clap::{crate_name, crate_version, App, Arg};
use rusty_dftd_lib::cutoff::{RealspaceCutoff, RealspaceCutoffBuilder};
use rusty_dftd_lib::defaults::CN_CUTOFF_D3_DEFAULT;
use rusty_dftd_lib::dftd3::damping3::{D3Param, D3ParamBuilder, RationalDamping3Param};
use rusty_dftd_lib::dftd3::disp3::get_dispersion as get_d3_dispersion;
use rusty_dftd_lib::dftd3::model3::D3Model;
use rusty_dftd_lib::disp::DispersionResult;
use rusty_dftd_lib::model::{get_molecule_frame, Molecule};

fn main() {
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("rusty DFT-D4 for empirical dispersion correction")
        .arg(
            Arg::new("xyz-file")
                .about("Sets the xyz file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("model")
                .about(
                    "Choose the dispersion model. \n\
                        Options:\n\
                        - GD3: Grimme D3-dispersion\n\
                        - GD4: Grimme D4-Dispersion",
                )
                .short('m')
                .long("model")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("charge")
                .about("Sets the total charge")
                .short('c')
                .long("charge")
                .required(false)
                .takes_value(true),
        )
        .arg(
            Arg::new("3-body.dispersion")
                .about("Turn on three body ATM dispersion")
                .long("atm")
                .required(false)
                .takes_value(false),
        )
        .arg(
            Arg::new("gradient")
                .about("Choose to calculate the gradient")
                .short('g')
                .long("gradient")
                .required(false)
                .takes_value(false),
        )
        .get_matches();

    let geom_file = matches.value_of("xyz-file").unwrap();

    let disp_model = matches.value_of("model").unwrap();

    let charge: i8 = if matches.is_present("charge") {
        matches.value_of("charge").unwrap().parse::<i8>().unwrap()
    } else {
        0
    };

    let atm = matches.is_present("atm");

    let grad = matches.is_present("gradient");

    let frame: Frame = get_molecule_frame(geom_file);
    let mut mol: Molecule = Molecule::from(frame);
    mol.set_charge(charge);

    let disp_result: DispersionResult = match disp_model.to_lowercase().as_str() {
        "gd3" => {
            let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new()
                .set_cn(CN_CUTOFF_D3_DEFAULT)
                .build();
            let disp: D3Model = D3Model::from_molecule(&mol, None);
            let d3param: D3Param = D3ParamBuilder::new()
                .set_s6(1.0000)
                .set_s8(0.2641)
                .set_s9(1.0000)
                .set_a1(0.0000)
                .set_a2(5.4959)
                .build();
            let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));
            get_d3_dispersion(&mut mol, &disp, &param, &cutoff, atm, grad, 4)
        }
        "gd4" => {
            panic!("D4-dispersion is not yet implemented!")
        }
        _ => {
            panic!("The dispersion method {} is not supported!", disp_model)
        }
    };

    if grad {
        let energy = disp_result.energy;
        let gradient = disp_result.gradient.unwrap();
        let sigma = disp_result.sigma.unwrap();

        println!("Dispersion energy:\n{:20.16}\n", energy);
        println!("Gradient:\n{:20.16}\n", gradient);
        println!("Sigma:\n{:20.16}\n", sigma);
    } else {
        let energy = disp_result.energy;

        println!("Dispersion energy:\n{:20.16}\n", energy);
    }
}

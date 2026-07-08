mod anneal;
mod perturbation;
mod eigentools;
mod ising;
use crate::anneal::*;
use crate::perturbation::*;
use crate::eigentools::*;
use crate::ising::*;
use faer::{Mat, c64};
use std::fs::File;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::path::Path;

enum Model{
  GraphInit{
    graph:Graph,
    temp: f64,
    x_vec: Vec<f64>,
    x0: Mat<c64>
  },
  IsingInit{
    ising:Ising,
    temp:f64,
  }
}

#[derive(Parser)]
#[command(about = "Toy Simulated Annealing Modeller")]
struct Args{
  input_file: PathBuf,
  #[arg(short = 'n', long = "steps", default_value="30")]
  steps: usize,
  #[arg(short = 'a', long = "alpha", default_value="0.95")]
  alpha: f64,
  #[arg(short = 'm', long = "model", default_value = "graph")]
  model: String,
  #[command(subcommand)]
  mode: Mode
}

#[derive(Subcommand)]
enum Mode{
  #[command(about = "Run experiment")]
  Experiment{
    #[arg(short = 's', long = "stationary_file", 
      default_value = "stationary_hits.txt")]
    stationary_file: PathBuf,
    #[arg(short = 'g', long = "geometric_file", 
      default_value = "geometric_hits.txt")]
    geometric_file: PathBuf,
    #[arg(short = 'N', long = "trials", 
      default_value="1000000")]
    trials:usize
  },
  #[command(about = "Show info about eigenvalues and vectors")]
  Eigeninfo{
    #[arg(short = 'f', long="eigen_file", default_value="eigen_evolution.txt")]
    eigen_file: PathBuf,
  },
  #[command(about = "Run perturbation series")]
  Perturbation{
    #[arg(short = 'o', long="order", default_value="1")]
    order:usize,
    #[arg(short = 's', long="stationary_file", 
      default_value="stationary_cum_prob.txt")]
    stationary_file:PathBuf,
    #[arg(short = 'p', long="perturbation_file", 
      default_value="perturbation_cum_prob.txt")]
    perturbation_file:PathBuf
  },
  #[command(about = "Output data necessary for computing parallel efficiency")]
  Parallel{
    #[arg(short = 'p', long = "target_probability", default_value="0.75")]
    target_p:f64
  },
  Lambda2{
    low_temp:f64,
    high_temp:f64,
    #[arg(short = 'f', long = "lambda_2_file", default_value="lambda_2.txt")]
    lambda_2_file:PathBuf
  }
}

impl Model{
  fn from_file(model_choice: &str, input_path: impl AsRef<Path>) -> Self {
    match model_choice {
      "graph" => {
      // Call your global or structural file-reader
        let (graph, temp, x_vec) = from_file(input_path);
        let x0 = Mat::<c64>::from_fn(x_vec.len(), 1, |i, _|
          c64::new(x_vec[i],0.0))
          .transpose()
          .to_owned();
        Model::GraphInit { graph, temp, x_vec, x0 }
      }
      "ising" => {
        let (ising, temp) = from_ising_file(input_path);
        Model::IsingInit { ising, temp }
      }
      unsupported => panic!(
        "Unknown model type: '{unsupported}'. 
        Only 'ising' and 'graph' are supported."
      ),
    }
  } 
}

fn main() {
  let args = Args::parse();
  let input_file = args.input_file;
  let steps = args.steps;
  let alpha = args.alpha;
  let model = Model::from_file(&args.model, input_file);


  match args.mode{
    Mode::Experiment{stationary_file, geometric_file, trials} =>{
    let mut geom_file = File::create(geometric_file).unwrap();
    let mut stationary_file = File::create(stationary_file).unwrap();
    
    match model{
      Model::GraphInit { graph, temp, x_vec, x0:_ } => {
        gather_data(
          &graph, 
          x_vec, 
          temp, 
          alpha, 
          trials, 
          &mut geom_file,
          &mut stationary_file
        );
      },
      Model::IsingInit { mut ising, temp } => {
        gather_ising_data(
          &mut ising,
          temp,
          alpha,
          trials,
          &mut geom_file,
          &mut stationary_file
        );
      }
    }
  },
  Mode::Perturbation { order, stationary_file, perturbation_file }=> {
      match model {
        Model::GraphInit{graph, temp, x_vec:_, x0} => {
    let mut stationary_theory_file = File::create(stationary_file)
      .unwrap();
    stationary_theory(temp, &mut stationary_theory_file, &graph, &x0, steps);
    let mut perturbation_theory_file = File::create(perturbation_file)
      .unwrap();
    perturbation(temp, alpha, &mut perturbation_theory_file, &graph, &x0, 
      steps, order);
        }
        Model::IsingInit{ising, temp} => {
          panic!("Perturbation not functional for Ising yet!");
        }
      }
  },
    Mode::Eigeninfo{ eigen_file } => {
      match model{
        Model::GraphInit { graph, temp, x_vec:_, x0 } => {
      let mut eigen_file = File::create(eigen_file)
        .unwrap();
      eigen_evolution(temp, alpha, &mut eigen_file, &graph, &x0, steps, None);
        }
        Model::IsingInit{ mut ising, temp } =>{
          let graph = ising.to_graph();
          let x_vec_ising = ising.start_configs_x_vec();
          let x0 :Mat<c64>;
          match &x_vec_ising{
            Some(x_vec) => {
              x0 = Mat::<c64>::from_fn(x_vec.len(), 1, |i, _|
                  c64::new(x_vec[i],0.0))
                .transpose()
                .to_owned();
          }
            None =>{
              assert!(ising.n_points <= 64, "Cannot do analysis on 
                ising with more than 64 nodes");
              x0 = Mat::<c64>::full(2usize.pow(ising.n_points as u32),
              1, c64::new((1/2usize.pow(ising.n_points as u32)) as f64, 0.0))
            }
          }
      let mut eigen_file = File::create(eigen_file)
        .unwrap();
      eigen_evolution(temp, alpha, &mut eigen_file, &graph, &x0, steps, None);
        }
      }
    },
    Mode::Parallel{ target_p } => {
      match model{
        Model::GraphInit { graph, temp, x_vec:_, x0 } => {
      let (alpha, k) = coarse_grain_search(temp, target_p, &graph, &x0);
      println!("Computed {} as optimal alpha after {} iterations", alpha, k)
        }
        Model::IsingInit{ising, temp} => {
          panic!("Mode not functional for Ising yet!");
    }
      }
    },
    Mode::Lambda2{ low_temp, high_temp, lambda_2_file }=> {
      match model{
        Model::GraphInit { graph, temp, x_vec:_, x0:_ } => {
      let mut out_file = File::create(lambda_2_file)
        .expect("Could not create file for lambda_2 info");
      lambda_2_range(high_temp, low_temp, &mut out_file, &graph, steps);
        }
        Model::IsingInit{ising, temp} => {
          panic!("Mode not functional for Ising yet!");
      }
        }
  }
  }
}

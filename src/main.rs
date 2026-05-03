mod anneal;
mod perturbation;
mod eigentools;
mod exact_solns;
use crate::anneal::*;
use crate::perturbation::*;
use crate::exact_solns::*;
use faer::{Mat, c64};
use std::fs::File;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Toy Simulated Annealing Modeller")]
struct Args{
  input_file: PathBuf,
  #[arg(short = 'n', long = "steps", default_value="30")]
  steps: usize,
  #[arg(short = 'a', long = "alpha", default_value="0.95")]
  alpha: f64,
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
    eigen_file: PathBuf
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

  }
}
  

fn main() {
  let args = Args::parse();
  let input_file = args.input_file;
  let steps = args.steps;
  let alpha = args.alpha;
  let (graph, temp, x_vec)=from_file(input_file);
  let x0 = Mat::<c64>::from_fn(x_vec.len(), 1, |i, _| c64::new(x_vec[i],0.0))
    .transpose()
    .to_owned();
  match args.mode{
    Mode::Experiment{stationary_file, geometric_file, trials} =>{
    let mut geom_file = File::create(geometric_file).unwrap();
    let mut stationary_file = File::create(stationary_file).unwrap();
    gather_data(&graph, x_vec, temp, alpha, trials, &mut geom_file, &mut stationary_file);
    },
    Mode::Perturbation { order, stationary_file, perturbation_file }=> {
    let mut stationary_theory_file = File::create(stationary_file)
      .unwrap();
    stationary_theory(temp, &mut stationary_theory_file, &graph, &x0, steps);
    let mut perturbation_theory_file = File::create(perturbation_file)
      .unwrap();
    perturbation(temp, alpha, &mut perturbation_theory_file, &graph, &x0, 
      steps, order);
    },
    Mode::Eigeninfo{ eigen_file } => {
      let mut eigen_file = File::create(eigen_file)
        .unwrap();
      eigen_evolution(temp, alpha, &mut eigen_file, &graph, &x0, steps);
    }
  }
}

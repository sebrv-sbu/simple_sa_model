use crate::anneal::*;
use crate::ising::*;
use faer::{Mat, c64};
use num_traits::Pow;

pub enum Model{
  GraphInit{
    graph:Graph,
    temp: f64,
    x_vec: Vec<f64>,
    x0: Mat<c64>
  },
  IsingInit{
    ising:Ising,
    temp:f64,
  },
}


impl Model{
  pub fn to_graph(&mut self)->&Self{
  match self{
    Model::GraphInit{ .. } => {},
    Model::IsingInit{ ising, temp } => {
      let graph = ising.to_graph();
      let x_vec = ising.start_configs_x_vec()
        .unwrap_or(
          vec![1.0/2.0_f64.pow(ising.n_points as f32); ising.n_points]
        );
      let x0 = Mat::<c64>::from_fn(x_vec.len(), 1, |i, _| 
        c64::new(x_vec[i], 0.0))
        .transpose()
        .to_owned();
      
      *self = Model::GraphInit{graph, temp: *temp, x_vec, x0}
      }
    }
  self
  }
}

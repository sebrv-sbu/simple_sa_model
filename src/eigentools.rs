use faer::{Mat, linalg::solvers::Solve, c64};
use crate::anneal::*;
use std::io::{BufWriter, Write};
use std::fs::File;

struct EigenDecomposition{
  mass_vector:Vec<c64>,
  lambda_vector:Vec<c64>,
  eigenvectors:Mat<c64>
}

pub trait OneNormalize{
  fn one_normalize(self) -> Self;
  fn one_normalize_inplace(&mut self);
}

impl OneNormalize for Mat<c64>{
  fn one_normalize(mut self) -> Self{
    self.one_normalize_inplace();
    self
  }
  fn one_normalize_inplace(&mut self){
    for j in 0..self.ncols(){
      let norm:f64 = self.col(j)
        .iter()
        .map(|x| x.norm())
        .sum::<f64>();

      self.col_mut(j)
        .iter_mut()
        .for_each(|x| *x = *x/c64::new(norm,0.0))
    }
  }
}

impl EigenDecomposition{
  pub fn eigendecompose(vector:&Mat<c64>, matrix:&Mat<c64>) -> Self{
    let eigen = matrix
      .to_owned()
      .eigen()
      .unwrap();
    let u = eigen
      .U()
      .to_owned()
      .one_normalize();


    let n = vector.nrows();
    let y = u
      .to_owned()
      .partial_piv_lu()
      .solve(&vector);
    
    let lambdas : Vec<c64> = eigen
      .S()
      .column_vector()
      .iter()
      .copied()
      .collect();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
      lambdas[b].norm().partial_cmp(&lambdas[a].norm()).unwrap()
    });

    let mass_vector:Vec<c64> = indices
      .iter()
      .map(|&i| 
        if u[(0,i)].re < 0.0 {
         -y[(i,0)]
        } else { y[(i,0)] }
      )
      .collect();
    let lambda_vector:Vec<c64> = indices
      .iter()
      .map(|&i| lambdas[i])
      .collect();
    let eigenvectors:Mat<c64> = Mat::from_fn(n, n, |row, col|{
      if u[(0,indices[col])].re < 0.0{
        -1.0 * u[(row, indices[col])]
      } else { u[(row, indices[col])] }
    });

    EigenDecomposition{mass_vector, lambda_vector, eigenvectors}
  }
}

pub fn eigen_evolution(init_temp:f64, alpha:f64, output_file: &mut File, 
  graph:&Graph, x0:&Mat<c64>, steps:usize){

  let mut buffed_output = BufWriter::new(output_file);
  let mut temp = init_temp;
  let mut curr_vec = x0.clone().transpose().to_owned();
  let mut curr_eigvecs: Option<Mat<c64>> = None;
  let mut prev_eigvecs: Option<Mat<c64>>;
 
  write!(buffed_output, "step").unwrap();
  for col in 0..curr_vec.nrows(){
    write!(buffed_output, "\teigval_{}\tmass_{}\tdiff_{}", col+1, col+1, col+1).unwrap();
    for row in 0..curr_vec.nrows(){
      write!(buffed_output,"\tvec{}_{}", col+1, row+1).unwrap();
    }
  }
  writeln!(buffed_output).unwrap();


  for i in 0..steps{
    let sa_hit_matrix = graph
      .to_hitting_matrix(temp, 0)
      .transpose()
      .to_owned()
      .map(|x| c64::new(*x, 0.0))
      .to_owned();
    let decomp = EigenDecomposition::eigendecompose(&curr_vec, &sa_hit_matrix);
    prev_eigvecs = curr_eigvecs.take();
    curr_eigvecs = Some(decomp.eigenvectors);
    if let Some(prev) = &prev_eigvecs {
      let curr = curr_eigvecs.as_ref().unwrap();
      let diff:Vec<f64> = (curr - prev)
        .col_iter()
        .map(|col| 
          col.iter()
          .map(|x| x.norm())
          .sum())
        .collect();
      write!(buffed_output, "{}", i).unwrap();
      for col in 0..curr.ncols() {
        write!(buffed_output, "\t{}\t{}\t{}",
          decomp.lambda_vector[col].norm(),
          decomp.mass_vector[col].re,
          diff[col]
        ).unwrap();
        for row in 0..curr.nrows() {
          write!(buffed_output, "\t{}", curr[(row, col)].re)
            .unwrap();
        }
      }
      writeln!(buffed_output).unwrap();
    } else {
      let curr = curr_eigvecs.as_ref().unwrap();
      write!(buffed_output, "{}", i).unwrap();
      for col in 0..curr.ncols() {
        write!(buffed_output, "\t{}\t{}\t{}",
          decomp.lambda_vector[col].norm(),
          decomp.mass_vector[col].re,
          "N/A",
        ).unwrap();
        for row in 0..curr.nrows() {
          write!(buffed_output, "\t{}", curr[(row, col)].re)
            .unwrap();
        }
      }
      writeln!(buffed_output).unwrap();
    }
    curr_vec = sa_hit_matrix * &curr_vec;
    temp *= alpha;
  }
}

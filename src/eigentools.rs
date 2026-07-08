use faer::{Mat, linalg::solvers::Solve, c64};
use crate::anneal::*;
use std::io::{BufWriter, Write};
use std::fs::File;
use num_traits::NumCast;

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

pub trait Return2ndEigInfo<T1: NumCast, T2:NumCast>{
  fn return_v_and_lambda_2(&self) -> (Mat<T1>, T2);
}

impl Return2ndEigInfo<c64, f64> for Mat<f64>{
  fn return_v_and_lambda_2(&self) -> (Mat<c64>, f64){
    let n = self.nrows();
    assert!(self.ncols() == n, "Did not supply a square matrix!");
    let eigs = self
      .to_owned()
      .map(|x| c64::new(*x, 0.0))
      .to_owned()
      .eigen()
      .expect("Could not find eigenvalues");
    let eigvals: Vec<c64> = eigs
      .S()
      .column_vector()
      .iter()
      .copied()
      .collect();

    let mut indices:Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
      eigvals[b].norm().partial_cmp(&eigvals[a].norm())
        .expect("Could not sort eigenvalues")
     });
    let lambda_2: f64 = eigvals[indices[1]].re;
    let v_2 = eigs
      .U()
      .col(indices[1]);
    (Mat::from_fn(v_2.nrows(), 1, |i, _| v_2[i]), lambda_2)
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

pub fn eigen_evolution(
  init_temp:f64,
  alpha:f64,
  output_file: &mut File, 
  graph:&Graph,
  x0:&Mat<c64>,
  steps:usize,
  global_min:Option<usize>)
{
  let mut buffed_output = BufWriter::new(output_file);
  let mut temp = init_temp;
  let mut curr_vec = x0.clone().transpose().to_owned();
  let mut curr_eigvecs: Option<Mat<c64>> = None;
  let mut prev_eigvecs: Option<Mat<c64>>;
  let sink = global_min.unwrap_or(graph.lowest_node());

  write!(buffed_output, "step").unwrap();
  for col in 0..curr_vec.nrows(){
    write!(buffed_output, "\teigval_{}\tmass_{}\tdiff_{}", col+1, col+1, col+1).unwrap();
    for row in 0..curr_vec.nrows(){
      write!(buffed_output,"\tvec{}_{}", col+1, row).unwrap();
    }
  }
  writeln!(buffed_output).unwrap();


  for i in 0..steps{
    let sa_hit_matrix = graph
      .to_hitting_matrix(temp, sink)
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


pub fn lambda_2_range(high_temp:f64, low_temp:f64, output_file: &mut File,
  graph:&Graph, n:usize) 
  {
  let delta_t:f64 = (high_temp - low_temp)/n as f64;
  let (v_2_inf, lambda_2_inf) = graph
    .to_hitting_matrix(f64::INFINITY, 0)
    .transpose()
    .to_owned()
    .return_v_and_lambda_2();
  let mut buffed_output = BufWriter::new(output_file);
  write!(buffed_output, "T\tlambda_2").expect("Could not write to output file");
  for row in 0..v_2_inf.nrows(){
    write!(buffed_output, "vec2_{}\t", row).expect("Could not write to output file");
  }
  writeln!(buffed_output).expect("Could not write to output file");
  write!(buffed_output, "{}\t{}", f64::INFINITY, lambda_2_inf)
    .expect("Could not write to output file");
  for row in 0..v_2_inf.nrows(){
    write!(buffed_output, "\t{}", v_2_inf[(row, 0)].re)
      .expect("Could not write to output file");
  }
  writeln!(buffed_output).expect("Could not write to output file");
  for i in 0..n{
    let temp = high_temp - (delta_t * i as f64);
    let (v_2, lambda_2) = graph
      .to_hitting_matrix(temp, 0)
      .transpose()
      .to_owned()
      .return_v_and_lambda_2();
    write!(buffed_output, "{}\t{}", temp, lambda_2)
      .expect("Could not write to output file");

    for row in 0..v_2.nrows(){
     write!(buffed_output, "\t{}", v_2[(row, 0)].re)
      .expect("Could not write to output file");
    }
    writeln!(buffed_output).expect("Could not write to output file");
  }
  let (v_2_0, lambda_2_0) = graph
    .to_hitting_matrix(0.0, 0)
    .transpose()
    .to_owned()
    .return_v_and_lambda_2();
  write!(buffed_output, "{}\t{}", 0.0, lambda_2_0)
    .expect("Could not write to output file");

  for row in 0..v_2_0.nrows(){
    write!(buffed_output, "\t{}", v_2_0[(row, 0)].re)
     .expect("Could not write to output file");
  }
  writeln!(buffed_output).expect("Could not write to output file");
}

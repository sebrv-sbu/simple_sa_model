use std::fs::File;
use faer::{Mat, c64};
use crate::anneal::*;
use crate::eigentools::*;
use std::io::{BufWriter, Write};

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

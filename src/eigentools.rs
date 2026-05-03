use faer::{Mat, linalg::solvers::Solve, c64};


pub struct EigenDecomposition{
  pub mass_vector:Vec<c64>,
  pub lambda_vector:Vec<c64>,
  pub eigenvectors:Mat<c64>
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

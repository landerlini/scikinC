/***************************************************************************/
/* File automatically generated with scikinC (github.com/landerli/scikinC) */
/*                                                                         */
/*                       D O   N O T   E D I T   ! ! !                     */
/*                                                                         */
/*  File generated on 2021-03-24 15:53                                     */
/*  by lucio                                                              */
/*  using MinMaxScalerConverter                          as converter      */
/*                                                                         */
/***************************************************************************/
#define FLOAT_T float

    extern "C" 
    FLOAT_T* a_minmaxscaler (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T input_min[] = {-4.09334020079585680918, -5.24084072667203315632, -0.32497285443652129677, -0.26355674574889492723, 3.12529579699870074805}; 
      FLOAT_T input_max[] = {5.76124430584669688926, 5.25341099097998665002, 0.34291674182486159284, 0.25861958732696688212, 5.27676285660240651509}; 
      FLOAT_T output_min = 0.000000; 
      FLOAT_T output_max = 1.000000; 

      for (int c = 0; c < 5; ++c)
        ret [c] = (input[c] - input_min[c]) / (input_max[c] - input_min[c]) 
                  * (output_max - output_min) + output_min;

      return ret;
    }
      

    extern "C" 
    FLOAT_T* a_minmaxscaler_inverse (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T input_min = 0.000000; 
      FLOAT_T input_max = 1.000000; 
      FLOAT_T output_min[] = {-4.09334020079585680918, -5.24084072667203315632, -0.32497285443652129677, -0.26355674574889492723, 3.12529579699870074805}; 
      FLOAT_T output_max[] = {5.76124430584669688926, 5.25341099097998665002, 0.34291674182486159284, 0.25861958732696688212, 5.27676285660240651509}; 

      for (int c = 0; c < 5; ++c)
        ret [c] = (input[c] - input_min) / (input_max - input_min) 
                  * (output_max[c] - output_min[c]) + output_min[c];

      return ret;
    }
      

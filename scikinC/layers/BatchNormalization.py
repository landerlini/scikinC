from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c


class BatchNormalization(BaseLayerConverter):
    """
    BatchNormalization Layer converter.

    According to the docs, this is an element-wise layer returning
    gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta
    """

    def definition(self):
        """Return the definition of the layer function"""
        ret = []

        gamma = self.layer.gamma.numpy()
        beta = self.layer.beta.numpy()
        mean = self.layer.moving_mean.numpy()
        var = self.layer.moving_variance.numpy()
        epsilon = self.layer.epsilon
        n_inputs = len(gamma)

        ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i;
            const FLOAT_T gamma[%(n_inputs)d] = %(gamma)s;
            const FLOAT_T beta[%(n_inputs)d] = %(beta)s;
            const FLOAT_T mean[%(n_inputs)d] = %(mean)s;
            const FLOAT_T var[%(n_inputs)d] = %(var)s;
            const FLOAT_T epsilon = %(epsilon)f;
            
            for (i = 0; i < %(n_inputs)d; ++i)
              ret[i] = gamma[i] * (input[i] - mean[i]) / sqrt(var[i] + epsilon) + beta[i];

            return ret; 
        }
        """ % dict(
            layername=self.name,
            n_inputs=n_inputs,
            epsilon=epsilon,
            gamma=array2c(gamma),
            beta=array2c(beta),
            mean=array2c(mean),
            var=array2c(var),
        )]

        return "\n".join(ret)

    def call(self, obuffer, ibuffer):
        """Return the call to the layer function"""
        return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict(
            layername=self.name, obuffer=obuffer, ibuffer=ibuffer
        )

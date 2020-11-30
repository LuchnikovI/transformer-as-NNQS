import tensorflow as tf
from samples_processing import hamiltonian
from math import pi


class AutoregressiveWaveFunction():
    """The class provides tools for representing a many-body wave function as
    an autoregressive model.
    
    Args: 
        model: a keras based autoregressive model that transforms
            a sequence of inputs to a sequence of the same shape,
            but shifted in time on an one time step."""

    def __init__(self, model):
        self.model = model

    @tf.function
    def sample(self, number_of_samples, chain_length):
        """Returns samples form |psi|^2.

        Args:
            number_of_samples: int value, number of samples
            chain_length: int value, number of subsystems in a
                whole system
        
        Return:
            int valued tensor of shape (number_of_samples, chain_length) filled
            by values of indices."""

        # initial sample
        smpl = tf.ones((number_of_samples, 1), dtype=tf.int32)
        # initial index
        i = tf.constant(0)
        # condition of tf.while_loop
        cond = lambda smpl, i: i < chain_length
        # body of tf.while_loop
        def body(smpl, i):
            # output logits
            out = self.model(smpl)
            local_dim = tf.cast(tf.shape(out)[-1] / 2, dtype=tf.int32)
            logits = tf.reshape(out, (number_of_samples, i + 1, 2, local_dim))[:, :, 0]
            logits = logits[:, -1]
            # new sample
            eps = tf.random.uniform((number_of_samples, local_dim))
            eps = -tf.math.log(-tf.math.log(eps))
            new_smpl = tf.argmax(tf.nn.log_softmax(logits) + eps, axis=-1,
                                 output_type=tf.int32)
            # adding new sample to the sequance of samples
            smpl = tf.concat([smpl, new_smpl[:, tf.newaxis]], axis=-1)
            return smpl, i + 1
        # loop
        smpl, _ = tf.while_loop(cond, body, loop_vars=[smpl, i],
                                shape_invariants=[tf.TensorShape((number_of_samples, None)), i.get_shape()])
        return smpl[:, 1:]

    @tf.function
    def value(self, samples):
        """Reutrns value of a wave function in a given point.

        Args:
            samples: int valued tensor of shape
                (number_of_samples, chain_length)

        Return:
            two real valued tensors of shape (number_of_samples,),
            the first one is log probability, the second one is phase"""

        number_of_samples = tf.shape(samples)[0]
        # output of the model
        smpl = tf.ones((number_of_samples, 1), dtype=tf.int32)
        out = self.model(tf.concat([smpl, samples], axis=-1))[:, :-1]
        # log probability and phase
        local_dim = tf.cast(tf.shape(out)[-1] / 2, dtype=tf.int32)
        out = tf.reshape(out, (number_of_samples, -1, 2, local_dim))
        logits, phi = out[:, :, 0], out[:, :, 1]
        one_hot_samples = tf.one_hot(samples, axis=-1, depth=local_dim)
        log_p = tf.reduce_sum(one_hot_samples * tf.nn.log_softmax(logits),
                              axis=(-2, -1))
        phi = tf.reduce_sum(one_hot_samples * pi * tf.nn.softsign(phi),
                            axis=(-2, -1))
        return log_p, phi

    def local_energy(self, h, samples, separation):
        '''Returns real and imag parts of local energy
        
        Args:
            h: list of two tensors representing many-body hamiltonian,
                the first tensor is an int valued tensot of shape
                (number_of_local_terms, chain_size), the second tensor is
                a real valued tensor of shape (number_of_local_terms,)
            samples: int valued tensor of shape
                (number_of_samples, chain_length)
            separation: tuple of two numbers that shows how we split
                a system into subsystems
        
        Return:
            two tensors of shape (number_of_samples,),
            real and imag parts of local energy'''

        size = separation[0]
        number_of_samples = tf.shape(samples)[0]
        # "passing" samples through a hamiltonian
        ravel_samples, ampls = hamiltonian(samples, h, separation)
        ravel_samples = tf.reshape(ravel_samples, (-1, size))
        # values of the nominator and denominator
        log_p_nom, phi_nom = self.value(ravel_samples)
        log_p_nom = tf.reshape(log_p_nom, (number_of_samples, -1))
        phi_nom = tf.reshape(phi_nom, (number_of_samples, -1))
        log_p_denom, phi_denom = self.value(samples)
        log_p_denom = log_p_denom[:, tf.newaxis]
        phi_denom = phi_denom[:, tf.newaxis]
        # local energy
        psi_ratio = tf.math.exp((log_p_nom - log_p_denom) / 2)
        dphi = phi_nom - phi_denom
        ampl_phi = tf.math.angle(ampls)
        ampl_modulus = tf.math.abs(ampls)
        imag = tf.reduce_sum(ampl_modulus * psi_ratio * tf.math.sin(dphi + ampl_phi), axis=-1)
        real = tf.reduce_sum(ampl_modulus * psi_ratio * tf.math.cos(dphi + ampl_phi), axis=-1)
        return real, imag

    @tf.function
    def observables_average(self, obs, samples, separation):
        '''Returns real and imag parts of averaged observables

        Args:
            obs: list of two tensors representing pauli strings
                with corresponding amplitudes, the first tensor
                is an int valued tensot of shape
                (number_of_observables, chain_size), the second tensor is
                a real valued tensor of shape (number_of_observables,)
            samples: int valued tensor of shape
                (number_of_samples, chain_length)
            separation: tuple of two numbers that shows how we split
                a system into subsystems

        Return:
            two tensors of shape (number_of_observables,),
            real and imag parts of observables'''

        size = separation[0]
        number_of_samples = tf.shape(samples)[0]
        # "passing" samples through an observable
        ravel_samples, ampls = hamiltonian(samples, obs, separation)
        ravel_samples = tf.reshape(ravel_samples, (-1, size))
        # values of the nominator and denominator
        log_p_nom, phi_nom = self.value(ravel_samples)
        log_p_nom = tf.reshape(log_p_nom, (number_of_samples, -1))
        phi_nom = tf.reshape(phi_nom, (number_of_samples, -1))
        log_p_denom, phi_denom = self.value(samples)
        log_p_denom = log_p_denom[:, tf.newaxis]
        phi_denom = phi_denom[:, tf.newaxis]
        # average values of pauli strings
        psi_ratio = tf.math.exp((log_p_nom - log_p_denom) / 2)
        dphi = phi_nom - phi_denom
        ampl_phi = tf.math.angle(ampls)
        ampl_modulus = tf.math.abs(ampls)
        imag = tf.reduce_mean(ampl_modulus * psi_ratio * tf.math.sin(dphi + ampl_phi), axis=0)
        real = tf.reduce_mean(ampl_modulus * psi_ratio * tf.math.cos(dphi + ampl_phi), axis=0)
        return real, imag

    @tf.function
    def grad(self, h, samples, separation, E):
        """Returns gradient of the state energy wrt parameters of a
        autoregressive model.

        Args:
            h: list of two tensors representing mny_body hamiltonian,
                the first tensor is an int valued tensot of shape
                (number_of_local_terms, chain_size), the second tensor is
                a real valued tensor of shape (number_of_local_terms,)
            samples: int valued tensor of shape
                (number_of_samples, chain_length)
            separation: tuple of two numbers that shows how we split
                a system into subsystems
            prev_E: float value representing baseline.
            """

        E_loc_real, E_loc_imag = self.local_energy(h, samples, separation)
        with tf.GradientTape() as tape:
            log_p, phi = self.value(samples)
            loss = 2 * tf.reduce_mean((log_p / 2) * (E_loc_real - E) + phi * E_loc_imag, axis=0)
        gradient = tape.gradient(loss, self.model.weights)
        return gradient, tf.reduce_mean(E_loc_real)
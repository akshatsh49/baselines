import tensorflow as tf
from tensorflow.keras import initializers
import edward2 as ed

def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_shape, num_actions):
        # the sub Functional model which does not include the top layer.
        model = network(input_shape)

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation=None)(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

        if dueling:
            with tf.name_scope("state_value"):
                state_out = latent
                for hidden in hiddens:
                    state_out = tf.keras.layers.Dense(units=hidden, activation=None)(state_out)
                    if layer_norm:
                        state_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(state_out)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.keras.layers.Dense(units=1, activation=None)(state_out)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return tf.keras.Model(inputs=model.inputs, outputs=[q_out])

    return q_func_builder

def build_q_func_multihead(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_shape, num_actions, n_heads):
        model = network(input_shape)    # sub Functional model (excluding top layer), backbone

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            head_action_outputs = [None] * n_heads
            for head in range(n_heads):
                action_out = latent
                for hidden in hiddens:
                    # action_out = tf.keras.layers.Dense(units=hidden, activation=None)(action_out)
                    action_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.VarianceScaling( ), bias_initializer = initializers.Zeros())(action_out)
                    # action_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.Zeros(), bias_initializer = initializers.Zeros())(action_out)
                    # action_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.GlorotUniform( ), bias_initializer = initializers.Zeros())(action_out)
                    if layer_norm:
                        action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                    action_out = tf.nn.relu(action_out)
                # action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)
                action_scores = tf.keras.layers.Dense(units=num_actions, activation=None, kernel_initializer = initializers.VarianceScaling( ), bias_initializer = initializers.Zeros())(action_out)
                head_action_outputs[head] = action_scores
            head_action_outputs = tf.convert_to_tensor(head_action_outputs)

        if dueling:
            head_state_outputs = [None] * n_heads
            for head in range(n_heads):
                with tf.name_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        # state_out = tf.keras.layers.Dense(units=hidden, activation=None)(state_out)
                        state_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.VarianceScaling( ), bias_initializer = initializers.Zeros())(state_out)
                        if layer_norm:
                            state_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(state_out)
                        state_out = tf.nn.relu(state_out)
                    # state_score = tf.keras.layers.Dense(units=1, activation=None)(state_out)
                    state_score = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer = initializers.VarianceScaling( ), bias_initializer = initializers.Zeros())(state_out)
                    head_state_outputs[head] = state_score

            head_state_outputs = tf.convert_to_tensor(head_state_outputs)
            action_scores_mean = tf.reduce_mean(head_action_outputs, 2)
            action_scores_centered = head_action_outputs - tf.expand_dims(action_scores_mean, 2)
            q_out = head_state_outputs + action_scores_centered
        else:
            q_out = head_action_outputs
        
        model = tf.keras.Model(inputs=model.inputs, outputs=[q_out])
        return model

    return q_func_builder

def build_q_func_sngp(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network, spectral_normalized = False)(**network_kwargs) #########################################################

    def q_func_builder(input_shape, num_actions):
        model = network(input_shape)
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]
        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                # action_out = ed.layers.SpectralNormalization(tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.VarianceScaling(), bias_initializer = initializers.Zeros()), norm_multiplier=0.9)(action_out)
                action_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer = initializers.VarianceScaling(), bias_initializer = initializers.Zeros())(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)

            if dueling:
                print(f"Dueling SNGP not implemented yet")
                raise NotImplementedError

            else :
                q_out = action_out
            
            # q_out = ed.layers.RandomFeatureGaussianProcess(num_actions, num_inducing=1024, gp_cov_momentum=0.99)(q_out)
            q_out = ed.layers.RandomFeatureGaussianProcess(num_actions, num_inducing=1024, gp_cov_momentum=0.99, kernel_initializer = tf.constant_initializer(250))(q_out)
        return tf.keras.Model(inputs=model.inputs, outputs=[q_out])

    return q_func_builder

# def build_q_func_sngp(input_size, num_actions, network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
#     class sngp_network(tf.keras.Model):
#         def __init__(self, input_size, num_actions, network, hiddens, dueling, layer_norm, **network_kwargs):
#             super().__init__()
#             self.input_size = input_size
#             self.num_actions = num_actions
#             self.network = network
#             self.hiddens = hiddens
#             self.dueling = dueling
#             self.layer_norm = layer_norm

#             if isinstance(self.network, str):
#                 from baselines.common.models import get_network_builder
#                 self.network = get_network_builder(self.network, spectral_normalized=True)(**network_kwargs)

#             self.model = self.network(input_size)    # sub Functional model (excluding top layer), backbone, EXPECTED TO CONTAIN SPECTRAL NORMALIZATION
#             self.flatten_layer = tf.keras.layers.Flatten()
#             self.action_layers_hidden = []
#             for hidden in self.hiddens:
#                 self.action_layers_hidden.append(ed.layers.SpectralNormalization(tf.keras.layers.Dense(units=hidden, activation=tf.nn.relu)))
#                 if layer_norm:
#                     self.action_layers_hidden.append(tf.keras.layers.LayerNormalization(center=True, scale=True))
#                 self.action_layers_hidden.append(tf.nn.relu)
            
#             if self.dueling:
#                 # define state head 
#                 print(f"Dueling SNGP not implemented yet")
#                 raise NotImplementedError

#             self.gp_head = ed.layers.RandomFeatureGaussianProcess(self.num_actions, num_inducing = 16, gp_cov_momentum=0.99)
        
#         def call(self, inputs, training = True, return_covmat = False):
#             x = self.flatten_layer(self.model(inputs))
#             for layer in self.action_layers_hidden:
#                 x = layer(x)
#             logits, covmat = self.gp_head(x)
            
#             if not training and return_covmat:
#                 return logits, covmat
#             return logits
        
#         ################## covariance resets pending here ##################
    
#     return sngp_network(input_size, num_actions, network, hiddens, dueling, layer_norm, **network_kwargs)

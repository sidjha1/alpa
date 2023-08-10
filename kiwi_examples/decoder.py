import alpa

import jax.numpy as jnp
import jax


batch_size = 16
seq_len = 1024
hidden_dim = 5120
num_layers = 1
heads = 40

main_rng = jax.random.PRNGKey(42)

params = {
    'Q_proj': [],
    'K_proj': [],
    'V_proj': [],
    'W1': [],
    'W2': [],
    'Q_proj_one': [],
    'K_proj_one': [],
    'V_proj_one': [],
    'head_proj': [],
    'head_proj_one': []
}

for _ in range(num_layers):
    main_rng, Q_rng = jax.random.split(main_rng)
    main_rng, K_rng = jax.random.split(main_rng)
    main_rng, V_rng = jax.random.split(main_rng)
    main_rng, W1_rng = jax.random.split(main_rng)
    main_rng, W2_rng = jax.random.split(main_rng)
    main_rng, Q_proj_one_rng = jax.random.split(main_rng)
    main_rng, K_proj_one_rng = jax.random.split(main_rng)
    main_rng, V_proj_one_rng = jax.random.split(main_rng)
    main_rng, head_proj_one_rng = jax.random.split(main_rng)
    main_rng, head_proj_rng = jax.random.split(main_rng)

    Q_proj = jax.random.uniform(Q_rng, shape=(hidden_dim, hidden_dim))
    K_proj = jax.random.uniform(K_rng, shape=(hidden_dim, hidden_dim))
    V_proj = jax.random.uniform(V_rng, shape=(hidden_dim, hidden_dim))
    W1 = jax.random.uniform(W1_rng, shape=(hidden_dim, 4 * hidden_dim))
    W2 = jax.random.uniform(W2_rng, shape=(4 * hidden_dim, hidden_dim))
    Q_proj_one = jax.random.uniform(
        Q_proj_one_rng, shape=(hidden_dim, hidden_dim))
    K_proj_one = jax.random.uniform(
        K_proj_one_rng, shape=(hidden_dim, hidden_dim))
    V_proj_one = jax.random.uniform(
        V_proj_one_rng, shape=(hidden_dim, hidden_dim))
    head_proj_one = jax.random.uniform(
        head_proj_one_rng, shape=(hidden_dim, hidden_dim))
    head_proj = jax.random.uniform(
        head_proj_rng, shape=(hidden_dim, hidden_dim))

    params['Q_proj'].append(Q_proj)
    params['K_proj'].append(K_proj)
    params['V_proj'].append(V_proj)
    params['W1'].append(W1)
    params['W2'].append(W2)
    params['Q_proj_one'].append(Q_proj_one)
    params['K_proj_one'].append(K_proj_one)
    params['V_proj_one'].append(V_proj_one)
    params['head_proj_one'].append(head_proj_one)
    params['head_proj'].append(head_proj)

mask = (jnp.triu(jnp.ones((seq_len, seq_len))) -
        jnp.eye(seq_len)) * float('-inf')
# Convert nans to 0s
mask = jnp.where(jnp.isnan(mask), 0, mask)

@alpa.parallelize
def alpa_train_step(parameters, batch):
        def loss_func(params, batch):
            input = batch['x']
            for i in range(num_layers):
                # Apply softmax
                # (batch_size, seq_len, hidden_dim) * (hidden_dim, hidden_dim) =
                # (batch_size, seq_len, hidden_dim)
                Q_one = jnp.matmul(input, params['Q_proj_one'][i])
                K_one = jnp.matmul(input, params['K_proj_one'][i])
                V_one = jnp.matmul(input, params['V_proj_one'][i])

                # (batch_size, seq_len, heads, hidden_dim / heads)
                Q_one = jnp.reshape(
                    Q_one, (batch_size, seq_len, heads, int(hidden_dim / heads)))
                K_one = jnp.reshape(
                    K_one, (batch_size, seq_len, heads, int(hidden_dim / heads)))
                V_one = jnp.reshape(
                    V_one, (batch_size, seq_len, heads, int(hidden_dim / heads)))

                # (batch_size, heads, seq_len, hidden_dim / heads)
                Q_one = jnp.transpose(Q_one, axes=(0, 2, 1, 3))
                K_one = jnp.transpose(K_one, axes=(0, 2, 1, 3))
                V_one = jnp.transpose(V_one, axes=(0, 2, 1, 3))

                # (batch_size, heads, seq_len, hidden_dim / heads) *
                # (batch_size, heads, hidden_dim / heads, seq_len) =
                # (batch_size, heads, seq_len, seq_len)
                attention_matrix = jnp.matmul(
                    Q_one, jnp.transpose(K_one, axes=(0, 1, 3, 2)))
                attention_matrix = attention_matrix * \
                    (1 / jnp.sqrt(hidden_dim))
                # Apply mask
                attention_matrix = attention_matrix + mask
                attention_matrix = jax.nn.softmax(attention_matrix)
                # (batch_size, heads, seq_len, seq_len) *
                # (batch_size, heads, seq_len, hidden_dim / heads) =
                # (batch_size, heads, seq_len, hidden_dim / heads)
                attention_output = jnp.matmul(attention_matrix, V_one)

                # (batch_size, seq_len, heads, hidden_dim / heads)
                attention_output = jnp.transpose(
                    attention_output, axes=(0, 2, 1, 3))

                # (batch_size, seq_len, hidden_dim)
                attention_output = jnp.reshape(
                    attention_output, (batch_size, seq_len, hidden_dim))
                attention_output = jnp.matmul(
                    attention_output, params['head_proj_one'][i])

                # Add + Layer Norm
                layer_norm_output = attention_output + input
                mean = jnp.mean(layer_norm_output, axis=2, keepdims=True)
                var = jnp.var(layer_norm_output, axis=2, keepdims=True)
                layer_norm_output = (
                    layer_norm_output - mean) / jnp.sqrt(var)

                Q = jnp.matmul(layer_norm_output, params['Q_proj'][i])
                K = jnp.matmul(layer_norm_output, params['K_proj'][i])
                V = jnp.matmul(layer_norm_output, params['V_proj'][i])

                # (batch_size, seq_len, heads, hidden_dim / heads)
                Q = jnp.reshape(
                    Q, (batch_size, seq_len, heads, int(hidden_dim / heads)))
                K = jnp.reshape(
                    K, (batch_size, seq_len, heads, int(hidden_dim / heads)))
                V = jnp.reshape(
                    V, (batch_size, seq_len, heads, int(hidden_dim / heads)))

                # (batch_size, heads, seq_len, hidden_dim / heads)
                Q = jnp.transpose(Q, axes=(0, 2, 1, 3))
                K = jnp.transpose(K, axes=(0, 2, 1, 3))
                V = jnp.transpose(V, axes=(0, 2, 1, 3))

                # (batch_size, heads, seq_len, hidden_dim / heads) *
                # (batch_size, heads, hidden_dim / heads, seq_len) =
                # (batch_size, heads, seq_len, seq_len)
                attention_matrix = jnp.matmul(
                    Q, jnp.transpose(K, axes=(0, 1, 3, 2)))
                attention_matrix = attention_matrix * \
                    (1 / jnp.sqrt(hidden_dim))
                attention_matrix = jax.nn.softmax(attention_matrix)
                # (batch_size, heads, seq_len, seq_len) *
                # (batch_size, heads, seq_len, hidden_dim / heads) =
                # (batch_size, heads, seq_len, hidden_dim / heads)
                attention_output = jnp.matmul(attention_matrix, V)

                # (batch_size, seq_len, heads, hidden_dim / heads)
                attention_output = jnp.transpose(
                    attention_output, axes=(0, 2, 1, 3))

                # (batch_size, seq_len, hidden_dim)
                attention_output = jnp.reshape(
                    attention_output, (batch_size, seq_len, hidden_dim))
                attention_output = jnp.matmul(
                    attention_output, params['head_proj'][i])

                # Add + Layer Norm
                layer_norm_output = attention_output + layer_norm_output
                mean = jnp.mean(layer_norm_output, axis=2, keepdims=True)
                var = jnp.var(layer_norm_output, axis=2, keepdims=True)
                layer_norm_output = (
                    layer_norm_output - mean) / jnp.sqrt(var)

                # FFN
                ffn_output = jnp.matmul(layer_norm_output, params['W1'][i])
                ffn_output = jax.nn.relu(ffn_output)
                ffn_output = jnp.matmul(ffn_output, params['W2'][i])

                # Add + Layer Norm
                layer_norm_output = ffn_output + layer_norm_output
                mean = jnp.mean(layer_norm_output, axis=2, keepdims=True)
                var = jnp.var(layer_norm_output, axis=2, keepdims=True)
                layer_norm_output = (
                    layer_norm_output - mean) / jnp.sqrt(var)

                input = layer_norm_output

            loss = jnp.linalg.norm(input - batch['y'])**2
            return loss

        grads = jax.grad(loss_func)(parameters, batch)

        for key in parameters:
            for i in range(num_layers):
                parameters[key][i] = parameters[key][i] - \
                    0.01 * grads[key][i]

        return parameters

main_rng, x_rng = jax.random.split(main_rng)
x = jax.random.uniform(x_rng, shape=(batch_size, seq_len, hidden_dim))

main_rng, y_rng = jax.random.split(main_rng)
y = jax.random.uniform(y_rng, shape=(batch_size, seq_len, hidden_dim))

batch = {
    'x': x,
    'y': y
}

print(alpa_train_step(params, batch))
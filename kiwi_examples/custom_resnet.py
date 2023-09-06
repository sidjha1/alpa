import alpa

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries import optimizers

key = jax.random.PRNGKey(42)

# init_fun, conv_net = stax.serial(
#     stax.Flatten,
#     stax.Dense(100),
#     stax.FanOut(2),
#     stax.parallel(
#         stax.serial(
#             stax.Dense(100),
#             stax.Relu
#         ),
#         stax.Identity
#     ),
#     stax.FanInSum,
#     stax.Dense(100),
#     stax.Relu,
#     stax.Dense(1000),
#     stax.Softmax
# )

# TODO: Add BatchNorm
init_fun, conv_net = stax.serial(
    stax.Conv(64, (7,7), padding="SAME"),
    stax.Relu,
    stax.MaxPool((3,3)),
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(64, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(64, (3,3), padding="SAME")
        ),
        stax.Identity
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(64, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(64, (3,3), padding="SAME")
        ),
        stax.Identity
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(128, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(128, (3,3), padding="SAME")
        ),
        stax.Conv(128, (1,1), padding="SAME")
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(128, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(128, (3,3), padding="SAME")
        ),
        stax.Identity
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(256, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(256, (3,3), padding="SAME")
        ),
        stax.Conv(256, (1,1), padding="SAME")
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(256, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(256, (3,3), padding="SAME")
        ),
        stax.Identity
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(512, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(512, (3,3), padding="SAME")
        ),
        stax.Conv(512, (1,1), padding="SAME")
    ),
    stax.FanInSum,
    stax.Relu,
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Conv(512, (3,3), padding="SAME"),
            stax.Relu,
            stax.Conv(512, (3,3), padding="SAME")
        ),
        stax.Identity
    ),
    stax.FanInSum,
    stax.Relu,
    stax.Flatten,
    stax.Dense(1000),
    stax.Softmax
)


_, params = init_fun(key, (8192, 3, 224, 224))

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

def loss(params, images, targets):
    preds = conv_net(params, images)
    return -jnp.sum(preds * targets)

def update(params, x, y, opt_state):
    value, grads = jax.value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    # return get_params(opt_state), opt_state, value
    return opt_state


@alpa.parallelize
def alpa_train_step(opt_state, batch):
    # params, opt_state, loss = update(get_params(opt_state), batch['x'], batch['y'], opt_state)
    opt_state = update(get_params(opt_state), batch['x'], batch['y'], opt_state)
    return opt_state


key, x_rng = jax.random.split(key)
x = jax.random.uniform(x_rng, shape=(8192, 3, 224, 224))

key, y_rng = jax.random.split(key)
y = jax.random.uniform(y_rng, shape=(8192, 1000))

batch = {
    'x': x,
    'y': y
}

print(alpa_train_step(opt_state, batch))
import alpa
from transformers import FlaxResNetForImageClassification
import jax
import os
import flax
import optax
from typing import Callable

import jax.numpy as jnp
from flax.training import train_state

os.environ['TRANSFORMERS_CACHE'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface'
os.environ['HF_HOME'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface'
os.environ['HUGGINGFACE_ASSETS_CACHE'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/rscratch/zhendong/lily/kiwi/.cache/huggingface/'

main_rng = jax.random.PRNGKey(42)

model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

learning_rate = 2e-5
class TrainState(train_state.TrainState):
    logits_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)

def adamw(weight_decay):
    return optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay)

def loss_function(logits, labels):
    return jnp.mean((logits[0] - labels) ** 2)

def eval_function(logits):
    return logits

state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=adamw(weight_decay=0.01),
    logits_function=eval_function,
    loss_function=loss_function,
)

@alpa.parallelize
def train_step(state, batch):
    targets = batch['y']

    def loss_function(params):
        logits = state.apply_fn(batch['x'], params=params, train=True)[0]
        loss = state.loss_function(logits, targets)
        return loss

    grad_function = jax.value_and_grad(loss_function)
    loss, grad = grad_function(state.params)
    new_state = state.apply_gradients(grads=grad)
    return new_state

x = jax.random.uniform(main_rng, shape=(1, 3, 256, 256))
y = jax.random.uniform(main_rng, shape=(1, 1000))

batch = {
    "x": x,
    "y": y
}

print(train_step(state, batch))
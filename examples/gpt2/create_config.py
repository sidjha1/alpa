from transformers import GPT2Config

config = GPT2Config.from_pretrained(
    "gpt2", 
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    vocab_size=50256,
    n_layer=1,
    cache_dir="/rscratch/zhendong/lily/kiwi/.cache/huggingface"
)

config.save_pretrained("./norwegian-gpt2")

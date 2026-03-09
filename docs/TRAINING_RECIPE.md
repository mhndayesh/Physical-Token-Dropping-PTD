# Physical Token Dropping (PTD): Training & Scaling Recipe

Because Physical Token Dropping (PTD) changes the fundamental flow of tensors by physically removing up to 70% of the sequence, you cannot drop it into a standard pre-training loop from scratch. The network will experience severe "shock" because the router will not know which tokens are mathematically important until it has learned basic language representations.

To scale this architecture successfully, we recommend a **Dual-Strategy Approach**: Distillation + Curriculum Sparsity, combined with a highly specific Data Curriculum.

---

## Strategy 1: The 3-Phase Curriculum

### Phase 1: The Teacher setup
Do not train PTD from scratch.
1. Start with a fully trained Dense model (e.g., Qwen 0.5B or 1.5B).
2. Inject the randomized PTD router into the architecture. At this point, the router's weights are untrained and its routing decisions are essentially random.

### Phase 2: Router Warm-up (Distillation)
Before we let PTD drop tokens, we need to teach the router how to route.
1. **Freeze the main model weights** (Attention and FFN).
2. **Keep retention at 100%** (or use soft-routing weights where tokens aren't dropped, but multiplied by the router's confidence score).
3. **Use the Dense Model as a Teacher:** Have both the original Dense model and your new PTD model process the same text. Train the router using *KL Divergence* — penalizing the PTD model if its output probability distribution deviates from the true dense model.
4. *Goal:* This teaches the router "Which tokens must I emphasize to make my output match the teacher?" before any information is actually destroyed.

### Phase 3: Curriculum Sparsity (The Squeeze)
Now we unfreeze the whole model and start dropping tokens, but we do it gently to avoid representation collapse.
1. **Start at 90% retention.** The model barely notices the dropped tokens.
2. **Slowly step down the retention rate** over thousands of steps: 90% → 70% → 50% → 30%.
3. *Goal:* As the sequence gets sparser, the FFN and Attention weights have time to gradually adjust to the missing context. The model learns to pack more semantic density into the tokens that *are* kept.

---

## Strategy 2: Mixed Data Curriculum

The training data mix is just as critical as the learning rate phase. The router learns by figuring out which tokens are "filler" and which carry dense logical weight.

- **General Mix:** Train on a standard, large-scale general corpus so the model doesn't overfit to one domain (WebText, Wikipedia, general conversational data).
- **Highly Selected "Dense" Data:** Mix in highly structured, information-dense data (Code, Math theorems, reasoning traces, logical puzzles).
- **The Synergy:** Mathematical and coding data trains the router to assign extreme high weights to syntax and logical operators (like `if`, `return`, `+`, `=`). General data trains the router to sift out conversational filler (like "uh," "so then," "anyway"). By mixing them, the router becomes a universal "meaning extractor."

---

## Strategy 3: Bring Your Own Recipe

If you have your own advanced curriculum or training loop (e.g., using DeepSeek-style RLExp, reinforcement learning, or dynamic routing curriculum), the basic rule remains the same:
- **Rule of Thumb:** Make sure the router has a way to learn *importance* before it's forced to act on it with physical token deletion. How you achieve that ramp-up is up to your specific engineering stack.

---

## The True Scaling Advantages

When you reach Phase 3 and the model is comfortably operating at 30% retention, your training infrastructure mathematics change entirely:

1. **Batch Size:** Because PTD uses 40-60% less VRAM (smaller activation tensors), you can double or triple your batch size on the exact same GPU hardware.
2. **Infinite Context Training:** Standard dense models hit an OOM wall quickly because attention scales quadratically (N²). Because PTD physically shrinks the sequence to `0.3N`, you can train on **32K or 64K context windows on consumer hardware.**
3. **Throughput:** With forward and backward passes running 3-4x faster, your step-time plummets. You can train on 300 billion tokens in the time it would previously take your cluster to train on 100 billion.

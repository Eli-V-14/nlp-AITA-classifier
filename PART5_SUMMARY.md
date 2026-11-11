# Prototype Implementation and Initial Evaluation

Project: predicting the verdict on an AITA post (NTA vs YTA) from the text of the story.

## 1. What we built

The whole pipeline runs end to end now: raw Reddit data in, cleaning and
labeling, training, and then either a prediction on a new post or an evaluation
on the test set.

On the data side (roadmap stages 1-3), `clean_aita.py` turns the raw dumps into
a labeled dataset. It throws out posts that were deleted or removed, strips the
EDIT/UPDATE sections and any verdict tags so the model can't cheat off them, and
maps each post's flair to a binary label (NAH folds into NTA, ESH into YTA). We
also keep the verdict counts from the top-level comments. After all the filtering
we're left with 373,340 labeled posts, and they're lopsided: about 76% NTA to
24% YTA.

For the classifier we fine-tuned roberta-base on 60k of those posts. We kept the
hyperparameters fixed instead of searching, since the papers we read said the
search isn't worth the compute. We did weight the loss by class frequency,
otherwise the model just learns to answer NTA every time. The trained model gets
saved to `out/model/`, so running inference or evaluation doesn't retrain
anything.

`infer.py` is the part a user actually interacts with. You give it a post (as an
argument, a file, or piped in) and it prints the verdict, the probability for
each class, and a line reminding you that this is only how Reddit would probably
vote, not an objective ruling.

What isn't done: the explanation generator. The plan was to fine-tune BART to
write a short comment explaining the verdict, and we haven't gotten there. So the
integration stage is only half built. The classifier side works start to finish;
the explanation side doesn't exist yet. We also used a fixed train/val/test split
for the transformer rather than cross-validation (we did run 5-fold CV on the
TF-IDF baseline).

## 2. How we evaluated it

It's binary classification, so accuracy is the obvious metric, but accuracy is
misleading here because the data is so imbalanced. A model that guesses NTA every
single time already scores about 81% on our test set. So the numbers we actually
pay attention to are macro-F1 and the per-class precision and recall, and we put
them next to a majority-class baseline and two TF-IDF baselines. This is exactly
the trap one of our papers (Efstathiadis et al.) fell into, so we were on the
lookout for it.

For test data we used the held-out test split (8,000 posts the model never saw
in training) plus 13 short cases we wrote by hand, phrased the way a person would
actually type a question rather than the way a Reddit post reads.

Natural test set (n = 8,000):

| Model | Accuracy | Macro-F1 | YTA-F1 |
|---|---:|---:|---:|
| Majority (all-NTA) | 0.815 | 0.449 | 0.000 |
| TF-IDF + Naive Bayes | 0.815 | 0.452 | 0.005 |
| TF-IDF + LogReg | 0.703 | 0.584 | 0.362 |
| roberta-base (fine-tuned) | 0.735 | **0.622** | 0.416 |

Cross-validated macro-F1 on the TF-IDF + LogReg model was 0.592 ± 0.003. On a
rebalanced 50/50 test set the fine-tuned model got 0.642 accuracy and 0.635
macro-F1. Confusion matrix (rows = true, columns = predicted, order NTA/YTA):
`[[5122, 1395], [728, 755]]`.

The thing worth explaining is that our model has *lower* accuracy than the
majority baseline (0.735 vs 0.815). That looks bad at first, but it's the whole
point. The baselines only win on accuracy because they almost never predict YTA,
which leaves their YTA-F1 at basically zero. Our model trades a bit of accuracy
to actually catch YTA cases (recall 0.51), and that's why it has the best
macro-F1 by a clear margin. The papers reported the same pattern.

On the 13 hand-written cases it got 12 right.

- It nailed the clear ones: telling your sister her cooking is terrible in front
  of her guests (YTA), refusing to give up a seat you paid for (NTA), eating a
  roommate's clearly labeled food (YTA).
- The one it missed: "WIBTA if I skipped my best friend's destination wedding
  because I can't afford the $3000 trip?" It said YTA; we'd call it NTA. It's a
  hypothetical built around wedding guilt, and the model seems to grab onto
  "skipping a friend's big event" and ignore the can't-afford-it part.

One caveat about all of these numbers: agreeing with Reddit isn't the same as
being morally right. We're measuring whether the model predicts how one online
crowd would vote, nothing more than that.

## 3. Problems we hit and what's next

The class imbalance is the main headache, and it shows up everywhere. It's why
accuracy is useless on its own, why we weight the loss, and why we test on a
rebalanced set too. The labels are also just noisy. A Reddit verdict is a crowd
opinion, not a fact, so there's a ceiling on how well any model can do here.

A data issue we ran into: the comment dump stops in March 2019 but the posts run
to mid-2022, so the newer posts (which is what the val and test splits are) have
labels but no comment signal. More on that in `preprocess/REPORT.md`. On the
tooling side, the machine was on Python 3.14, which PyTorch doesn't support yet,
so we set up a separate 3.12 environment for the model code.

Where the model is weakest right now is YTA: recall is 0.51 and precision is
0.35, so it's still missing or over-calling roughly half of them. And again,
there's no explanation component yet.

For the final version, roughly in order:
1. Build the explanation generator and connect it to the classifier.
2. Get YTA detection up. Train on more of the 300k posts, tune the decision
   threshold, and handle posts longer than 256 tokens.
3. Add the soft-label / vote-proportion evaluation using the counts we already
   stored.
4. Put a simple interface on the whole thing.

Artifacts: the saved model is in `out/model/`, the metrics are in
`out/eval_report.json` and `out/baseline_report.json`, and the top-level
`README.md` covers how to run everything.

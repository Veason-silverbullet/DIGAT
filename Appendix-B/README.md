# Supplementary Experiments on SA Strategy (Appendix B)
In Appendix B of the paper, we validate the effectiveness of semantic-augmentation (SA) strategy. It is worth noting that SA Strategy can be applied to general neural news recommendation models.

For NRMS reinforced with SA strategy, experiements on MIND-small and MIND-large:
<pre><code>python main.py --dataset=MIND-small --model=NRMS-SA
python main.py --dataset=MIND-large --model=NRMS-SA</code></pre>

The model prediction file for MIND-large is at `prediction/MIND-large/NRMS-SA/#1/prediction.zip`. After training, the `prediction.zip` can be subumitted to MIND leaderboard for performance evluation.

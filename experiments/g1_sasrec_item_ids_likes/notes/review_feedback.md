you need to implement mu Transfer
best combination for metrics is unreadable. Make separate tables for each hypothesis (one for dims, one for pos etc.). also it seems that it is better to put it after rqs since a lot of answers are answered in the different rqs.
No need for epoch time, peack memory, params etc in all rqs: it is needed in rq3 only. Also I mostly need epoch time vs recall there but other metrics are usefull too.
ffn_gelu looks strange. It seems that you are not tuning hyperparameters in some cases when hyperparameters tuning is needed. I understand that it will quite a bit of time, but if it is needed to make a conclusion it is needed (add it to instructions too). You can tune hyperparameters on the subset and do final training on the full dataset.
lr inverse sqrt does not look correct.
Also 4 runs is not probably enough to tune hyperparameters.
rq5 - you need to try more schedulers. Also it would be cool if you would "hand-tune" the lr scheduler for comparison: lets say epoch 5 is finished. Then change lr to 5 variants (some smaller, some larger and the same; probably with linear change or something like that), choose the best one and retrain epoch 6 with it. Then do the same for epoch 7. And so on (start from epoch 1). If it is hard to do, do not do it. But I think it can be achieved using callback on epoch end or something like that.
rq6 - your result about plain lr_warmup is incorrect. You most definitely did something wrong.
rq7 - did you try learnable pos + rope? learnable pos reverse + rope? learnable pos reverse + rope reverse? And other variants? If not, you need to try at least some of them (may be you can make a conclusion that adding reverse rope does not improve metrics => no need to try that; you need to state that in the report).
you need to try more variants of normalization. There is also batch norm, rmsnorm and may be some other.
decrease in recall after introduction of bos token is wrong. There is some sort of bug.
r19 you need to run it
rq10 you need to run this too.

other things:
also try different learning rate for embeddings and deep part.
report deep params separately from the embeddings params.

Your suggestions:
rq11 looks good
rq16 That is not a research question :) That needs to be done for hyperparameters tuning. If utransfer works, for some trainings you will not need to tune hyperparameters (mainly in scaling experiments by deep parameters - you'll be able to use utransfer)
rq12 - ok.
rq14 - sounds good. Also compare offline logq and online logq
rq15. I don't understand this one. Lets get back to it afterwards.
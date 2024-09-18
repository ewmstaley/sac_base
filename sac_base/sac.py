'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from sac_base.worker import SACWorker
from sac_base.timekeeper import TimeKeeper

def SAC(
        env_fn,
        policy_fn, 
        q_fn,
        training_steps=1e6,
        logs_dir="./logs",
        save_dir="./saved_models",
        checkpoint_every=None,
        **worker_kwargs
    ):

    tk = TimeKeeper(training_steps)

    worker = SACWorker(
        env_fn, 
        policy_fn, 
        q_fn,
        logs_dir=logs_dir, 
        save_dir=save_dir, 
        **worker_kwargs
    )

    worker.start_collection()

    while worker.total_steps < training_steps:

        # data collect
        worker.rollout_to_next_update()
        
        for j in range(worker.update_every):
            # get a batch
            worker.sample_latest_batch()

            # one grad step for Q
            worker.compute_grad_q()
            worker.update_q_networks()

            # one grad step for pi, with Q frozen (internal)
            worker.compute_loss_pi()
            worker.update_pi_networks() # unfreezes Q
            worker.update_alpha()

            # update target network (polyak)
            worker.polyak_q()

        worker.save() # save if new best performance

        if worker.total_steps%1000 == 0:
            print("Total steps:", worker.total_steps)
            tk.log(worker.total_steps)
            tk.report()

        # also save checkpoint every N steps
        if checkpoint_every is not None and worker.total_steps % checkpoint_every==0:
            worker.save(explicit_file="ckpt_"+str(worker.total_steps)+".pt")
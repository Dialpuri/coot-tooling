#!/bin/bash

SESH="coot-tooling"

tmux has-session -t $SESH 2>/dev/null

if [ $? != 0 ]; then 
    tmux new-session -d -s $SESH -n "batch"

    tmux send-keys -t $SESH:batch "cd /lmb/home/jdialpuri/Development/coot-tooling" C-m
    tmux send-keys -t $SESH:batch "source .venv/bin/activate" C-m
    tmux send-keys -t $SESH:batch "python -m tooling.batch coot --agent --mmdb-only" C-m
    #tmux send-keys -t $SESH:batch "python batch_cli.py --model gemma4:31b --probe-ptoolingb /lmb/home/jdialpuri/Development/coot-dev/coot/reference-structures/1c7k.pdb"

    tmux new-window -t $SESH -n "srun"
    tmux send-keys -t $SESH:srun "srun --partition=ml --gres=gpu:2 --time=72:00:00 --pty bash" C-m
    tmux send-keys -t $SESH:srun "OLLAMA_MODELS=/net/nfs6/gmssd/jdialpuri/.ollama OLLAMA_CONTEXT_LENGTH=90000 /lmb/home/jdialpuri/bin/ollama serve" C-m

    tmux new-window -t $SESH -n "tunnel"
    tmux send-keys -t $SESH:tunnel "ssh -L 11434:localhost:11434 \$(squeue --me -o "%N" | tail -1)" C-m 

    tmux select-window -t $SESH:batch

fi 

tmux attach-session -t $SESH

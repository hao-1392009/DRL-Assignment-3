import argparse
import pathlib
import os
import pickle
import collections
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gym.wrappers import LazyFrames
import lz4.block

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import FeatureExtrator, InverseModel
import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%m-%d %H:%M:%S")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame_size", type=int, nargs="*")
    parser.add_argument("--stack_frames", type=int)
    parser.add_argument("--lz4_compress", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--steps_per_log", type=int)
    parser.add_argument("--steps_per_save", type=int)
    parser.add_argument("--seed", type=int)

    return parser

def conver_data(raw_data, stack_frames, lz4_compress, frame_size):
    data = []

    frame_stack = collections.deque(maxlen=stack_frames)
    for observation, action in raw_data:
        if lz4_compress:
            observation = np.frombuffer(
                lz4.block.decompress(observation), dtype=np.float32
                ).reshape(*frame_size)

        if action is None:  # environment reset
            for _ in range(stack_frames):
                frame_stack.append(observation)
            state = LazyFrames(list(frame_stack), lz4_compress)
        else:
            frame_stack.append(observation)
            next_state = LazyFrames(list(frame_stack), lz4_compress)
            data.append((state, next_state, action))

            state = next_state

    return data

class FEInvDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        state, next_state, action = self.dataset[index]

        return (
            torch.FloatTensor(np.asarray(state)),
            torch.FloatTensor(np.asarray(next_state)),
            action
        )

    def __len__(self):
        return len(self.dataset)

def eval_and_log(feature_extractor, inverse_model, eval_loader, device,
                 epoch, train_loss, num_eval_data):
    feature_extractor.eval()
    inverse_model.eval()

    sum_eval_loss = 0
    for state, next_state, action in eval_loader:
        state, next_state, action = state.to(device), next_state.to(device), action.to(device)

        with torch.no_grad():
            feat_state = feature_extractor(state)
            feat_next_state = feature_extractor(next_state)

            prediction = inverse_model(torch.cat((feat_state, feat_next_state), 1))

            loss = F.cross_entropy(prediction, action.long(), reduction="sum")
            sum_eval_loss += loss.item()

    logger.info(
        f"Epoch: {epoch:.2f}"
        f", training loss: {train_loss}"
        f", evaluation loss: {sum_eval_loss / num_eval_data}"
    )

    feature_extractor.train()
    inverse_model.train()

def save_checkpoint(checkpoint_dir, feature_extractor, inverse_model, optimizer):
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    torch.save(
        feature_extractor.state_dict(),
        checkpoint_dir / "feature_extractor.pt"
    )
    torch.save(
        inverse_model.state_dict(),
        checkpoint_dir / "inverse_model.pt"
    )
    torch.save(
        optimizer.state_dict(),
        checkpoint_dir / "training_state.pt"
    )
    logger.info(f"Successfully saved checkpoint to {checkpoint_dir}")


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    if len(args.frame_size) == 1:  # type a single number in command line
        args.frame_size *= 2


    output_dir = pathlib.Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Training config: {vars(args)}")


    data_dir = pathlib.Path(args.data_dir)
    with open(data_dir / "train_data.pkl", "rb") as file:
        train_data = pickle.load(file)
    train_data = conver_data(train_data, args.stack_frames, args.lz4_compress, args.frame_size)

    with open(data_dir / "eval_data.pkl", "rb") as file:
        eval_data = pickle.load(file)
    eval_data = conver_data(eval_data, args.stack_frames, args.lz4_compress, args.frame_size)

    train_dataset = FEInvDataset(train_data)
    eval_dataset = FEInvDataset(eval_data)

    logger.info(f"Loaded {len(train_dataset)} training data")
    logger.info(f"Loaded {len(eval_dataset)} evaluation data")

    train_loader = DataLoader(train_dataset, args.train_batch_size, True)
    eval_loader = DataLoader(eval_dataset, args.eval_batch_size)


    util.fix_random_seed(args.seed)

    feature_extractor = FeatureExtrator((args.stack_frames, *args.frame_size))
    feature_size = feature_extractor.output_size
    inverse_model = InverseModel(feature_size * 2, 12)

    feature_extractor.to(args.device)
    inverse_model.to(args.device)
    feature_extractor.train()
    inverse_model.train()

    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(inverse_model.parameters()),
        args.learning_rate
    )

    train_losses = []
    step_count = 0

    logger.info("***** Start training *****")
    for epoch in range(args.num_epochs):
        for index, (state, next_state, action) in enumerate(train_loader):
            state, next_state, action = state.to(args.device), next_state.to(args.device), action.to(args.device)

            feat_state = feature_extractor(state)
            feat_next_state = feature_extractor(next_state)

            prediction = inverse_model(torch.cat((feat_state, feat_next_state), 1))

            loss = F.cross_entropy(prediction, action.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_losses.append(loss.item())
            step_count += 1

            if step_count % args.steps_per_log == 0:
                eval_and_log(
                    feature_extractor, inverse_model, eval_loader, args.device,
                    epoch + (index + 1) / len(train_loader),  # may not be accurate
                    np.mean(train_losses),
                    len(eval_dataset)
                )
                train_losses = []

            if step_count % args.steps_per_save == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step_count}"
                save_checkpoint(checkpoint_dir, feature_extractor, inverse_model, optimizer)

    if step_count % args.steps_per_log != 0:
        eval_and_log(
            feature_extractor, inverse_model, eval_loader, args.device,
            args.num_epochs,
            np.mean(train_losses),
            len(eval_dataset)
        )

    if step_count % args.steps_per_save != 0:
        checkpoint_dir = output_dir / f"checkpoint-{step_count}"
        save_checkpoint(checkpoint_dir, feature_extractor, inverse_model, optimizer)


if __name__ == "__main__":
    main()

# Custom nnU-Net trainers

nnU-Net resolves a trainer by CLASS NAME at load time, and it only searches its own
package. The checkpoint records `nnUNetTrainerFinetune1e3_500ep`, so without this
directory on `nnUNet_extTrainer` every prediction dies with

    RuntimeError: Could not find requested nnunet trainer nnUNetTrainerFinetune1e3_500ep
    in nnunetv2.training.nnUNetTrainer

`cysts.common.paths` sets `nnUNet_extTrainer` to this folder on import, so a fresh
install works without anyone having to know that.

The class only changes the learning rate and epoch count. Those matter for TRAINING and
are inert at inference -- but the name is still required to rebuild the network.

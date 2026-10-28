import os, pickle
from typing import Optional, Any
import numpy as np
import torch
import torch.nn as nn
import smplx
from smplx.lbs import lbs
from smplx.utils import Struct, to_tensor, MANOOutput
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from .rotations import axis_angle_to_matrix


class MANORotMatDecoder(smplx.SMPL):
    """
    MANO model that only supports rotation matrix pose parameters.
    (Edited from the official MANO implementation smplx.MANO)

    * The model includes extra joint indices (extra_joints_idxs) to incorporate fingertip joints, resulting in a total of 21 joints.
    * It reorders the joints using an OpenPose joint map (openpose_joint_map) to ensure consistency.
    * The calculate_wrist_offset() function allows for more precise coordinate system transformations by providing translation offset.
    """

    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        create_hand_pose: bool = True,
        hand_pose: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        dtype: Any = torch.float32,
        vertex_ids: Optional[dict] = None,
        use_compressed: bool = True,
        ext: str = "pkl",
    ) -> None:
        self.is_rhand = is_rhand
        if data_struct is None:
            if os.path.isdir(model_path):
                model_fn = "MANO_{}.{ext}".format("RIGHT" if is_rhand else "LEFT", ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if "RIGHT" in os.path.basename(model_path) else False
            assert os.path.exists(mano_path), "Path {} does not exist!".format(mano_path)
            if ext == "pkl":
                with open(mano_path, "rb") as mano_file:
                    model_data = pickle.load(mano_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)
        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]
        super(MANORotMatDecoder, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
        )
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS["mano"].values()), dtype=torch.long
        )
        hand_pose_dim = 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)
            hand_pose_param = nn.Parameter(default_hand_pose, requires_grad=True)
            self.register_parameter("hand_pose", hand_pose_param)
        openpose_joint_map = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]
        self.register_buffer("joint_map", torch.tensor(openpose_joint_map, dtype=torch.long))

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        hand_pose: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
    ) -> MANOOutput:
        assert global_orient is not None
        betas = betas if betas is not None else self.betas
        if hand_pose is None:
            hand_pose = self.hand_pose.view(-1, 3)
            hand_pose = axis_angle_to_matrix(hand_pose)[None]
        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        betas = betas.to(self.v_template.dtype)
        full_pose = full_pose.to(self.v_template.dtype)
        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_map is not None:
            joints = joints[:, self.joint_map, :]
        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)
        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )
        return output

    def calculate_wrist_offset(
        self, betas: Optional[torch.Tensor], global_orient: torch.Tensor, hand_pose: torch.Tensor
    ):
        betas = betas if betas is not None else self.betas
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        assert full_pose.shape[-1] == 6
        full_pose = rotation_6d_to_matrix(d6=full_pose)
        betas = betas.to(self.v_template.dtype)
        full_pose = full_pose.to(self.v_template.dtype)
        _, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )
        wrist_offset = joints[..., 0, :]
        return wrist_offset

    def name(self) -> str:
        return "MANORotMat"

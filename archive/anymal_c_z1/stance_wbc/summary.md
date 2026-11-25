robot=anymal_c_z1  torch mass plant=1.2 kg, controller model=1.2 kg

| scenario | fell | base_pos_dev_max_mm | base_rot_dev_max_deg | base_vel_rms_mm_s | base_angvel_rms_deg_s | torch_err_rms_mm | torch_err_max_mm | foot_slip_max_mm | peak_torque_frac | seam_ik_residual_mm | qp_ms_mean | qp_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weld | False | 0.401 | 0.0413 | 0.192 | 0.0595 | 0.168 | 0.608 | 0.0619 | 0.684 | 0.00184 | 0.21 | 0 |
| stress | False | 0.387 | 0.0221 | 0.199 | 0.0489 | - | - | 0.118 | 0.445 | 0.00184 | 0.177 | 0 |

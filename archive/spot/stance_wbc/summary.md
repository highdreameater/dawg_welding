robot=spot  torch mass plant=1.2 kg, controller model=1.2 kg

| scenario | fell | base_pos_dev_max_mm | base_rot_dev_max_deg | base_vel_rms_mm_s | base_angvel_rms_deg_s | torch_err_rms_mm | torch_err_max_mm | foot_slip_max_mm | peak_torque_frac | seam_ik_residual_mm | qp_ms_mean | qp_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weld | False | 0.287 | 0.00351 | 0.00326 | 0.000537 | 0.0152 | 0.0231 | 0.222 | 0.724 | 0.00464 | 0.17 | 0 |
| stress | False | 0.291 | 0.0131 | 0.0378 | 0.0231 | - | - | 0.807 | 0.828 | 0.00464 | 0.131 | 0 |

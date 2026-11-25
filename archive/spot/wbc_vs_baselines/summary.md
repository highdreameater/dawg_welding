torch mass plant=1.2 kg, controller model=1.2 kg

| scenario | controller | fell | base_sag_before_arm_mm | seam_unreachable | base_pos_dev_max_mm | base_rot_dev_max_deg | base_vel_rms_mm_s | base_angvel_rms_deg_s | torch_err_rms_mm | torch_err_max_mm | foot_slip_max_mm | peak_torque_frac | qp_ms_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weld | wbc | False | 39.7 | False | 0.287 | 0.00351 | 0.00326 | 0.000537 | 0.0152 | 0.0231 | 0.222 | 0.724 | 0.137 |
| weld | pd_soft | False | 227 | True | 13.1 | 2.09 | 0.413 | 0.221 | 299 | 299 | 2.64 | 0.544 | - |
| weld | pd_stiff | False | 35.1 | False | 6.81 | 0.743 | 2.21 | 0.11 | 12.9 | 16.2 | 17 | 0.539 | - |
| weld | rl_stand | False | 93.5 | False | 97.3 | 28.2 | 22.1 | 3.14 | 464 | 524 | 29.8 | 0.714 | - |
| stress | wbc | False | 39.7 | False | 0.291 | 0.0147 | 0.0382 | 0.0237 | - | - | 0.845 | 0.855 | 0.126 |
| stress | pd_soft | True | 227 | True | 1.06e+03 | 180 | 331 | 77.4 | - | - | 1.78e+03 | 1 | - |
| stress | pd_stiff | False | 35.1 | False | 13.9 | 2.08 | 32.3 | 3.92 | - | - | 15.4 | 0.556 | - |
| stress | rl_stand | True | 93.5 | False | 978 | 180 | 189 | 35.5 | - | - | 1.69e+03 | 1 | - |

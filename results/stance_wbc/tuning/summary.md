Stance WBC tuning study, robot=go2_z1_official

| variant | scenario | fell | base_pos_dev_max_mm | base_rot_dev_max_deg | base_vel_rms_mm_s | torch_err_rms_mm | torch_err_max_mm | foot_slip_max_mm | peak_torque_frac | qp_failures |
|---|---|---|---|---|---|---|---|---|---|---|
| tuned | weld | False | 0.398 | 0.0302 | 0.167 | 0.082 | 0.176 | 0.137 | 0.349 | 0 |
| v0_defaults | weld | False | 0.429 | 0.0478 | 0.299 | 1.46 | 3.18 | 0.139 | 0.351 | 0 |
| no_friction_comp | weld | False | 1.01 | 0.326 | 1.28 | 0.686 | 1.81 | 0.169 | 0.361 | 0 |
| friction_ff_only | weld | False | 0.474 | 0.0889 | 0.433 | 1.16 | 1.98 | 0.142 | 0.353 | 0 |
| no_armature | weld | False | 0.588 | 0.257 | 0.303 | 0.865 | 1.5 | 3.23 | 1 | 0 |
| rate_100hz | weld | False | 3.53 | 3.64 | 18.8 | 5.22 | 8.91 | 213 | 1 | 0 |
| rate_50hz | weld | False | 15.6 | 10.6 | 44.8 | 40.9 | 58.1 | 295 | 1 | 0 |
| rate_25hz | weld | True | 1.12e+03 | 180 | 620 | 861 | 1.15e+03 | 1.77e+03 | 1 | 11 |
| wrong_tool_mass | weld | False | 1.06 | 0.59 | 0.217 | 0.152 | 0.347 | 0.814 | 0.768 | 0 |
| soft_base | weld | False | 5 | 0.392 | 0.286 | 0.0815 | 0.173 | 0.364 | 0.35 | 0 |
| very_soft_base | weld | False | 14.9 | 0.561 | 0.504 | 0.0814 | 0.172 | 0.893 | 0.361 | 0 |
| underdamped_base | weld | False | 0.214 | 0.0519 | 0.347 | 0.0824 | 0.178 | 0.145 | 0.35 | 0 |
| very_underdamped | weld | False | 0.739 | 0.56 | 0.402 | 0.0825 | 0.179 | 0.134 | 0.35 | 0 |
| tool_3kg_unmodelled | weld | False | 1.66 | 1.39 | 0.602 | 0.189 | 0.448 | 8.59 | 1 | 0 |
| weak_base_weight | weld | False | 0.509 | 0.0304 | 0.175 | 0.0819 | 0.176 | 0.133 | 0.349 | 0 |
| no_contact_stab | weld | False | 0.412 | 0.0368 | 0.167 | 0.082 | 0.176 | 0.138 | 0.349 | 0 |
| too_stiff_base | weld | False | 0.0206 | 0.0119 | 0.44 | 0.0827 | 0.176 | 0.193 | 0.416 | 0 |
| tuned | stress | False | 1.5 | 0.798 | 2.84 | - | - | 0.443 | 0.377 | 0 |
| v0_defaults | stress | False | 1.5 | 0.798 | 2.84 | - | - | 0.443 | 0.377 | 0 |
| no_friction_comp | stress | False | 3.35 | 1.81 | 4.31 | - | - | 0.518 | 0.385 | 0 |
| friction_ff_only | stress | False | 1.5 | 0.798 | 2.84 | - | - | 0.443 | 0.377 | 0 |
| no_armature | stress | False | 1.45 | 0.905 | 2.72 | - | - | 0.422 | 0.375 | 0 |
| rate_100hz | stress | False | 1.59 | 0.813 | 3.18 | - | - | 0.452 | 0.387 | 0 |
| rate_50hz | stress | False | 1.64 | 0.793 | 3.97 | - | - | 0.465 | 0.373 | 0 |
| rate_25hz | stress | False | 1.72 | 0.776 | 7.69 | - | - | 0.524 | 0.448 | 0 |
| wrong_tool_mass | stress | False | 1.85 | 0.772 | 3.63 | - | - | 1.2 | 0.462 | 0 |
| soft_base | stress | False | 8.72 | 3.18 | 7.58 | - | - | 1.07 | 0.444 | 0 |
| very_soft_base | stress | False | 20.9 | 5.41 | 11.6 | - | - | 1.77 | 0.478 | 0 |
| underdamped_base | stress | False | 3.18 | 1.94 | 12.9 | - | - | 2.24 | 0.531 | 0 |
| very_underdamped | stress | False | 14.2 | 3.92 | 97.1 | - | - | 55.9 | 1 | 0 |
| tool_3kg_unmodelled | stress | False | 12.5 | 0.795 | 16.6 | - | - | 50.6 | 0.852 | 0 |
| weak_base_weight | stress | False | 1.53 | 0.802 | 2.86 | - | - | 0.449 | 0.375 | 0 |
| no_contact_stab | stress | False | 1.54 | 0.807 | 2.87 | - | - | 0.45 | 0.377 | 0 |
| too_stiff_base | stress | False | 0.166 | 0.0657 | 1.24 | - | - | 0.7 | 0.641 | 0 |

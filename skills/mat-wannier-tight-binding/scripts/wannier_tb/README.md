# Shared Wannier90 analysis

`io.py` validates standard folded `_hr.dat` / `_wsvec.dat` pairs, counted
fractional k-points, band-separated two-column data and `.win` cells. It also
keeps each command's parameters in a separate stage of `input_configs.yaml`.

`hamiltonian.py` applies both degeneracy factors and the orbital-pair shifts:

`H_mn(k) = sum_R H_mn(R)/N_R * sum_T exp(2*pi*i*k.(R+T))/N_T`.

It checks Hermiticity before removing rounding noise. Files with expanded,
already-weighted R lists must not be combined with an original folded sidecar;
the readers reject mismatched mappings. The skill uses the folded format from
Wannier90 v4.0.3. See the main skill for supported scope and commands.

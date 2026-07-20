
from __future__ import annotations

from .a_perception import A1, A2, A3, A4, A5, A6, A7, A8
from .b8_structural_growth import B8
from .b_biology import B1, B2, B3, B4, B5, B6, B7, B9, B10
from .base import Experiment
from .c_cogsci import C1, C2, C3, C4, C5, C6, C7, C8, C9
from .d6_sensitive_window import D6
from .d_developmental import D1, D2, D3, D4, D5, D7, D8, D9
from .e2_replay import E2
from .e3_plasticity import E3
from .e4_neuromod import E4
from .e5_curiosity import E5
from .e6_relational import E6
from .e7_sparse import E7
from .e8_dendritic import E8
from .e9_local import E9
from .e10_openended import E10
from .ex1_generative_replay import EX1
from .ex2_latent_planning import EX2
from .ex3_test_time_adaptation import EX3
from .ex4_fast_weights import EX4
from .ex5_local_rules_scale import EX5
from .ex6_active_inference import EX6
from .ex7_meta_learning import EX7
from .ex8_curiosity_bakeoff import EX8
from .ex9_slot_attention import EX9
from .ex10_cross_modal import EX10
from .ex11_causal_probing import EX11
from .ex12_atlas import EX12
from .ex13_long_stream import EX13
from .ex14_memory_bakeoff import EX14
from .ex15_rejuvenation import EX15
from .ex16_codebook_sr import EX16
from .ex17_latent_reasoning import EX17
from .ex18_self_verification import EX18
from .f_form_substrate import F1, F2, F3, F4, F5, F9, F10, F12, F13, F14, F17, F18, F19, F20
from .f_form_substrate_missing import F6, F7, F8, F11, F15, F16
from .i_infotheory import I1, I2, I3, I5, I6, I7, I8, I9, I4i
from .n_neuro_replay_reasoning import N1, N3, N4, N5, N6, N7, N8, N9, N10, N11
from .p_philosophy import P1, P2, P3, P4, P5, P6, P9, P10
from .s_semiotics import S1, S3, S4, S5, S6, S7, S9, S10
from .y4_hysteresis import Y4
from .y_dynamics import Y1, Y2, Y3, Y5, Y6, Y7, Y8, Y9

SCAFFOLDS: list[type[Experiment]] = [
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E8,
    E9,
    E10,
    EX1,
    EX2,
    EX3,
    EX4,
    EX5,
    EX6,
    EX7,
    EX8,
    EX9,
    EX10,
    EX11,
    EX12,
    EX13,
    EX14,
    EX15,
    EX16,
    EX17,
    EX18,
    N1,
    N3,
    N4,
    N5,
    N6,
    N7,
    N8,
    N9,
    N10,
    N11,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P9,
    P10,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    I1,
    I2,
    I3,
    I4i,
    I5,
    I6,
    I7,
    I8,
    I9,
    Y1,
    Y2,
    Y3,
    Y4,
    Y5,
    Y6,
    Y7,
    Y8,
    Y9,
    S1,
    S3,
    S4,
    S5,
    S6,
    S7,
    S9,
    S10,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
]

# Substrate interference report

Counterfactual update effects per declared parameter group, bounded short horizon, scored on the
tuning split only. Effects are lower 95 percent confidence bounds over 8 seeds and two domain
directions.

Read the old domain loss column carefully. For a domain local group it is exactly zero by
construction, because those parameters are not read by the other domain's forward pass, so no
forgetting was measured for them. Only the two shared rows carry a measured forgetting number.

| group | new domain gain | old domain loss | return recovery | verdict |
| --- | --- | --- | --- | --- |
| G.adapter | 0.2847 | 0.0 | 0.8226 | domain local, acquires; zero forgetting is structural not measured |
| G.head | 0.2646 | 0.0 | 0.8233 | domain local, acquires; zero forgetting is structural not measured |
| G.norm | 0.2667 | 0.0 | 0.8234 | domain local, acquires; zero forgetting is structural not measured |
| G.proj_conv | 0.4411 | 0.0 | 0.8224 | domain local, acquires; zero forgetting is structural not measured |
| G.proj_lin | 0.39 | 0.0 | 0.8229 | domain local, acquires; zero forgetting is structural not measured |
| G.shared_fast_core | 0.4214 | 0.272 | 0.8175 | shared, acquisition bought with forgetting |
| H.adapter | 0.1716 | 0.0 | 0.8181 | domain local, acquires; zero forgetting is structural not measured |
| H.head | 0.1537 | 0.0 | 0.8161 | domain local, acquires; zero forgetting is structural not measured |
| H.proj_conv | 0.3267 | 0.0 | 0.8169 | domain local, acquires; zero forgetting is structural not measured |
| H.proj_lin | 0.2853 | 0.0 | 0.8197 | domain local, acquires; zero forgetting is structural not measured |
| H.shared_fast_delta | 0.262 | 0.2315 | 0.8204 | shared, acquisition bought with forgetting |

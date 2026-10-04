# Fixed-seed attack evaluation

The synthetic attack evaluation uses seeds 739001 through 739040 and six
reference truths [25,115,205,295,385,475]. Targets are sampled uniformly on
[0,500] from each fixed source seed. There are nine complete source packets,
gain [0.7,1.3], epsilon=5, and at most two controlled source identities.

Both exact and heterogeneous reference profiles use costs [1,2,1,2,1,2] and
budget 4. The attack library is defined in attack_evaluation_protocol.json and
its nested original_attack_protocol object. Target shifts, reference spoofing,
reference reversal, and whole-packet translations preserve all honest rows.
The 40 seeds are statistical units; repeated conditions within a seed are paired.
This synthetic experiment is distinct from the public 28-day primary evaluation.

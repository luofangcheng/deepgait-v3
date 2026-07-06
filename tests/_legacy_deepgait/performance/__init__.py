"""Performance / regression benchmark tests.

Acceptance gates (docs/DEVELOPMENT_PLAN.md §9.2):
  * 4 cameras x 100 fps @ 4MP full resolution (MV-CL042-10GM)
  * gait_algorithms: 1000 frames < 5 s (Cython)
  * DLC inference < 50 ms/frame
  * H5 write (key info only) < 50 MB/s

Mark every benchmark with @pytest.mark.slow so CI can deselect them:
    pytest -m "not slow"
"""

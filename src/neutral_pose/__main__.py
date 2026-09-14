"""Allow ``python -m neutral_pose specimen.zip`` as an alias of ``neutral-pose-auto``."""

from .auto import main

raise SystemExit(main())

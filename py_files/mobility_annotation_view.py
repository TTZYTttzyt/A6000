from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class MobilityAnnotationView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "MobilityAnnotationView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._drawers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/mobility_annotation/link_1", name="drawers_view", reset_xform_properties=False
        )
        self._handles = RigidPrimView(
            prim_paths_expr="/World/envs/.*/mobility_annotation/link_4", name="handles_view", reset_xform_properties=False
        )
        self._doors = RigidPrimView(
            prim_paths_expr="/World/envs/.*/mobility_annotation/link_0", name="doors_view", reset_xform_properties=False
        )
        #/World/envs/.*/partnet_3cdabe258ed67a144da5feafe6f1c8fc/link_1

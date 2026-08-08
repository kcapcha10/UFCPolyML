"""Feature engine — protocol contracts and domain models.

Point-in-time feature engineering for UFC fight prediction. Implements strict
temporal isolation via event-atomic replay with emit-before-update semantics.
Every feature depends only on information strictly prior to the fight.
"""

from ufc_edge.features.contracts import (
    EmitContext as EmitContext,
)
from ufc_edge.features.contracts import (
    EventTick as EventTick,
)
from ufc_edge.features.contracts import (
    FeatureEmitter as FeatureEmitter,
)
from ufc_edge.features.contracts import (
    FeatureRow as FeatureRow,
)
from ufc_edge.features.contracts import (
    FighterProfile as FighterProfile,
)
from ufc_edge.features.contracts import (
    FightOutcomeView as FightOutcomeView,
)
from ufc_edge.features.contracts import (
    FightTotals as FightTotals,
)
from ufc_edge.features.contracts import (
    FrozenState as FrozenState,
)
from ufc_edge.features.contracts import (
    HistoricalFight as HistoricalFight,
)
from ufc_edge.features.contracts import (
    StateComponent as StateComponent,
)

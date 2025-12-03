"""
Scaffold for modeling a human body in code.

Design goals:
- Highly modular: body → systems → organs → tissues/cells.
- Extensible: easy to plug in new systems, organs, models (biomechanics, metabolism, etc.).
- Simulation-ready: clear hooks for time-stepping and interactions.
- Strong typing and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Any
from enum import Enum


# ================================================================
# CORE INTERFACES AND BASE CLASSES
# ================================================================

class TimeStepContext(Protocol):
    """
    Context object passed into each update step.
    Extend this with environmental conditions (temperature, oxygen, etc.).
    """

    # This interface is intentionally minimal; add as needed.
    @property
    def dt_seconds(self) -> float:
        ...


@dataclass
class SimpleTimeStepContext:
    """
    Basic implementation of TimeStepContext for simple simulations.
    """
    dt_seconds: float


class Updatable(ABC):
    """
    Anything that changes over time should implement this interface.
    """

    @abstractmethod
    def step(self, ctx: TimeStepContext) -> None:
        """
        Advance internal state by ctx.dt_seconds.
        """
        ...


class EnergyConsumer(Protocol):
    """
    Protocol for subsystems that consume energy (ATP, calories, etc.).
    """

    def energy_demand_watts(self) -> float:
        """
        Return power demand in Watts at the current state.
        """
        ...


class CirculationSink(Protocol):
    """
    Something that receives blood or other fluid.
    """

    def receive_flow(self, volume_ml: float, oxygen_fraction: float) -> None:
        """
        Receive a fluid flow from circulation.
        """
        ...


class SignalTarget(Protocol):
    """
    Target for neural or hormonal signaling.
    """

    def receive_signal(self, signal_name: str, magnitude: float) -> None:
        """
        Receive a signal from nervous or endocrine systems.
        """
        ...


# ================================================================
# BASIC SUPPORT STRUCTURES (ENUMS, VECTORS, ETC.)
# ================================================================

class Sex(Enum):
    """
    Simplified sex enumeration.
    """
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    UNDEFINED = "undefined"


@dataclass
class Vector3:
    """
    Minimal 3D vector for positions (e.g., joint locations).
    """
    x: float
    y: float
    z: float


@dataclass
class VitalSigns:
    """
    Container for high-level vital signs.
    """
    heart_rate_bpm: float = 70.0
    respiratory_rate_bpm: float = 12.0
    systolic_bp_mmHg: float = 120.0
    diastolic_bp_mmHg: float = 80.0
    body_temperature_c: float = 37.0
    oxygen_saturation_pct: float = 98.0
    blood_glucose_mg_dl: float = 90.0
    hydration_pct: float = 100.0


# ================================================================
# ORGAN AND TISSUE ABSTRACTIONS
# ================================================================

@dataclass
class Organ(Updatable):
    """
    Abstract organ representation.
    """
    name: str
    mass_kg: float
    position: Optional[Vector3] = None

    def step(self, ctx: TimeStepContext) -> None:
        """
        Default organ step does nothing.
        Override in specific organs.
        """
        # Intentionally empty; subclasses can override
        return


@dataclass
class MetabolicOrgan(Organ, EnergyConsumer):
    """
    Base class for organs with explicit metabolic demand.
    """
    basal_metabolic_rate_watts: float = 0.0

    def energy_demand_watts(self) -> float:
        """
        Return the organ's current energy demand.
        Can be overridden to depend on activity level.
        """
        # For now, return the basal rate
        return self.basal_metabolic_rate_watts


# ================================================================
# CIRCULATORY SYSTEM
# ================================================================

@dataclass
class BloodCompartment:
    """
    Simple compartment representing a volume of blood.
    """
    volume_ml: float
    oxygen_fraction: float  # 0.0–1.0

    def extract_oxygen(self, fraction: float) -> float:
        """
        Extract a fraction of the oxygen content and return the amount removed.
        """
        # Clamp fraction to [0,1]
        fraction = max(0.0, min(1.0, fraction))
        # Compute current oxygen content (arbitrary units)
        current_o2 = self.oxygen_fraction * self.volume_ml
        removed_o2 = current_o2 * fraction
        remaining_o2 = current_o2 - removed_o2
        # Update internal oxygen fraction
        self.oxygen_fraction = remaining_o2 / self.volume_ml if self.volume_ml > 0 else 0.0
        return removed_o2


@dataclass
class Heart(MetabolicOrgan):
    """
    Simple heart model with heart rate and stroke volume.
    """
    heart_rate_bpm: float = 70.0
    stroke_volume_ml: float = 70.0

    def compute_cardiac_output(self) -> float:
        """
        Return cardiac output in mL/s.
        """
        beats_per_second = self.heart_rate_bpm / 60.0
        return beats_per_second * self.stroke_volume_ml

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update heart state. For now, this is a placeholder.
        """
        # Example placeholder: could adapt rate based on signals or blood pressure
        return


@dataclass
class CirculatorySystem(Updatable):
    """
    Very high-level circulatory system.
    """

    heart: Heart
    arterial: BloodCompartment
    venous: BloodCompartment
    sinks: List[CirculationSink] = field(default_factory=list)
    pulmonary_oxygen_fraction: float = 0.98  # Lungs oxygenate blood toward this

    def register_sink(self, sink: CirculationSink) -> None:
        """
        Register a tissue or organ as a circulation sink.
        """
        self.sinks.append(sink)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Advance circulation by one time step.
        """
        # Step heart state first
        self.heart.step(ctx)

        # Calculate volume of blood pumped this step
        flow_ml_per_s = self.heart.compute_cardiac_output()
        volume_pumped = flow_ml_per_s * ctx.dt_seconds

        # Clamp pumped volume to available arterial blood
        volume_pumped = min(volume_pumped, self.arterial.volume_ml)

        # Compute per-sink allocation
        if self.sinks and volume_pumped > 0:
            per_sink_volume = volume_pumped / len(self.sinks)
        else:
            per_sink_volume = 0.0

        # Send blood to each sink
        for sink in self.sinks:
            sink.receive_flow(
                volume_ml=per_sink_volume,
                oxygen_fraction=self.arterial.oxygen_fraction,
            )

        # Move used blood from arterial to venous compartment
        self.arterial.volume_ml -= volume_pumped
        self.venous.volume_ml += volume_pumped

        # Re-oxygenate a portion of venous blood via lungs
        self._oxygenate_venous_blood(ctx)

    def _oxygenate_venous_blood(self, ctx: TimeStepContext) -> None:
        """
        Move a fraction of venous blood through lungs and into arterial compartment.
        """
        # Simple model: same flow as cardiac output
        flow_ml_per_s = self.heart.compute_cardiac_output()
        volume_lung = flow_ml_per_s * ctx.dt_seconds
        volume_lung = min(volume_lung, self.venous.volume_ml)

        if volume_lung <= 0:
            return

        # Compute oxygenated mixture
        incoming_o2 = self.pulmonary_oxygen_fraction * volume_lung

        # Remove blood from venous pool
        self.venous.volume_ml -= volume_lung

        # Assume venous blood has some current oxygen content
        venous_o2 = self.venous.oxygen_fraction * self.venous.volume_ml
        # (For a more precise model, track oxygen per volume. Here we simplify.)

        # Add lung-oxygenated blood directly to arterial compartment
        total_o2 = (self.arterial.oxygen_fraction * self.arterial.volume_ml) + incoming_o2
        self.arterial.volume_ml += volume_lung
        self.arterial.oxygen_fraction = total_o2 / self.arterial.volume_ml


# ================================================================
# RESPIRATORY SYSTEM
# ================================================================

@dataclass
class Lung(MetabolicOrgan, CirculationSink):
    """
    Simple lung model that exchanges gases with environment.
    """
    tidal_volume_ml: float = 500.0
    breaths_per_min: float = 12.0
    ambient_o2_fraction: float = 0.21

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update lung ventilation.
        """
        # Example placeholder: could track CO2, O2, and adjust ventilation
        return

    def receive_flow(self, volume_ml: float, oxygen_fraction: float) -> None:
        """
        Receive venous blood. In a full model, exchange O2/CO2 here.
        """
        # This method would connect to detailed gas-exchange models.
        return


@dataclass
class RespiratorySystem(Updatable):
    """
    Collection of organs responsible for breathing and gas exchange.
    """
    lungs: Lung

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update respiratory organs.
        """
        self.lungs.step(ctx)


# ================================================================
# NERVOUS SYSTEM
# ================================================================

@dataclass
class Neuron:
    """
    Simple neuron scaffold (not a realistic biophysical model).
    """
    name: str
    membrane_potential_mV: float = -70.0
    threshold_mV: float = -55.0
    refractory: bool = False

    def integrate(self, input_current: float, dt: float) -> bool:
        """
        Integrate input current and return True if neuron fires.
        """
        if self.refractory:
            # Example: simple refractory behavior
            self.membrane_potential_mV = -70.0
            self.refractory = False
            return False

        # Integrate input (this is a placeholder; no biophysics)
        self.membrane_potential_mV += input_current * dt

        # Check for spike
        if self.membrane_potential_mV >= self.threshold_mV:
            self.refractory = True
            self.membrane_potential_mV = 30.0  # spike peak (placeholder)
            return True

        return False


@dataclass
class NeuralNetwork(Updatable):
    """
    Very abstract collection of neurons representing a brain or ganglion.
    """
    neurons: Dict[str, Neuron] = field(default_factory=dict)

    def add_neuron(self, neuron: Neuron) -> None:
        """
        Add a new neuron to the network.
        """
        self.neurons[neuron.name] = neuron

    def stimulate(self, neuron_name: str, current: float, dt: float) -> bool:
        """
        Apply a current to a named neuron and return whether it fired.
        """
        neuron = self.neurons.get(neuron_name)
        if neuron is None:
            return False
        return neuron.integrate(current, dt)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update network state.
        """
        # For now, do nothing globally. This is where you would
        # implement network dynamics, sensory input, and motor output.
        return


@dataclass
class NervousSystem(Updatable):
    """
    Nervous system with central (brain) and peripheral components.
    """
    central: NeuralNetwork
    signal_targets: Dict[str, SignalTarget] = field(default_factory=dict)
    basal_metabolic_rate_watts: float = 20.0

    def register_target(self, name: str, target: SignalTarget) -> None:
        """
        Register a target that can receive neural signals.
        """
        self.signal_targets[name] = target

    def send_signal(self, target_name: str, signal_name: str, magnitude: float) -> None:
        """
        Send a signal to a registered target.
        """
        target = self.signal_targets.get(target_name)
        if target is not None:
            target.receive_signal(signal_name, magnitude)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update nervous system state.
        """
        self.central.step(ctx)

    def energy_demand_watts(self) -> float:
        """
        Approximate metabolic demand of nervous tissue.
        """
        return self.basal_metabolic_rate_watts


# ================================================================
# MUSCULOSKELETAL SYSTEM
# ================================================================

@dataclass
class Muscle(MetabolicOrgan, SignalTarget, CirculationSink):
    """
    Simple muscle that can be activated by neural signals.
    """
    activation: float = 0.0  # 0.0–1.0
    oxygen_debt: float = 0.0

    def receive_signal(self, signal_name: str, magnitude: float) -> None:
        """
        Receive a signal and adjust activation.
        """
        if signal_name == "contract":
            # Clamp activation to [0, 1]
            self.activation = max(0.0, min(1.0, self.activation + magnitude))
        elif signal_name == "relax":
            self.activation = max(0.0, self.activation - magnitude)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update muscle state (fatigue, force, etc.).
        """
        # Example placeholder: decay activation slightly over time
        decay_rate = 0.1
        self.activation = max(0.0, self.activation - decay_rate * ctx.dt_seconds)

    def receive_flow(self, volume_ml: float, oxygen_fraction: float) -> None:
        """
        Receive blood flow and update simple oxygen debt tracking.
        """
        # Very coarse model: reduce oxygen debt based on available oxygen and activation.
        # In a richer simulation, this would feed into metabolism and force production.
        oxygen_available = volume_ml * oxygen_fraction
        oxygen_use = min(oxygen_available, self.activation * 10.0)
        self.oxygen_debt = max(0.0, self.oxygen_debt - oxygen_use)


@dataclass
class Bone:
    """
    Minimal bone representation.
    """
    name: str
    length_m: float
    mass_kg: float
    proximal: Optional[Vector3] = None
    distal: Optional[Vector3] = None


@dataclass
class Joint:
    """
    Minimal joint connecting two bones.
    """
    name: str
    bone_a: Bone
    bone_b: Bone
    angle_deg: float = 0.0
    min_angle_deg: float = -180.0
    max_angle_deg: float = 180.0

    def set_angle(self, angle_deg: float) -> None:
        """
        Set the joint angle, clamped to allowed range.
        """
        self.angle_deg = max(self.min_angle_deg, min(self.max_angle_deg, angle_deg))


@dataclass
class MusculoskeletalSystem(Updatable):
    """
    Very high-level musculoskeletal system.
    """
    bones: Dict[str, Bone] = field(default_factory=dict)
    joints: Dict[str, Joint] = field(default_factory=dict)
    muscles: Dict[str, Muscle] = field(default_factory=dict)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update all muscles (and possibly kinematics).
        """
        for muscle in self.muscles.values():
            muscle.step(ctx)

        # Here you could update joint angles based on muscle forces.


# ================================================================
# ENDOCRINE SYSTEM
# ================================================================

@dataclass
class Hormone:
    """
    Simplified hormone representation for signaling between systems.
    """
    name: str
    concentration_ng_ml: float
    half_life_s: float

    def decay(self, ctx: TimeStepContext) -> None:
        """
        Exponentially decay hormone concentration over time.
        """
        if self.concentration_ng_ml <= 0:
            return
        decay_constant = 0.693 / max(self.half_life_s, 1e-6)
        self.concentration_ng_ml *= max(0.0, 1.0 - decay_constant * ctx.dt_seconds)


@dataclass
class EndocrineSystem(Updatable):
    """
    Simple endocrine scaffold tracking circulating hormones.
    """
    hormones: Dict[str, Hormone] = field(default_factory=dict)
    basal_metabolic_rate_watts: float = 5.0

    def release(self, name: str, amount_ng_ml: float, half_life_s: float = 60.0) -> None:
        """
        Add or increase a hormone level in circulation.
        """
        hormone = self.hormones.get(name)
        if hormone is None:
            hormone = Hormone(name=name, concentration_ng_ml=0.0, half_life_s=half_life_s)
            self.hormones[name] = hormone
        hormone.concentration_ng_ml += amount_ng_ml

    def level(self, name: str) -> float:
        """
        Retrieve current concentration for a hormone, if present.
        """
        hormone = self.hormones.get(name)
        return hormone.concentration_ng_ml if hormone else 0.0

    def step(self, ctx: TimeStepContext) -> None:
        """
        Decay hormone levels each timestep.
        """
        for hormone in self.hormones.values():
            hormone.decay(ctx)

    def energy_demand_watts(self) -> float:
        """
        Minimal metabolic cost for hormonal regulation.
        """
        return self.basal_metabolic_rate_watts


# ================================================================
# HEPATIC SYSTEM
# ================================================================

@dataclass
class Liver(MetabolicOrgan, CirculationSink):
    """
    Liver scaffold for detoxification and glycogen storage.
    """
    glycogen_store_kcal: float = 300.0
    glycogen_capacity_kcal: float = 600.0

    def receive_flow(self, volume_ml: float, oxygen_fraction: float) -> None:
        """
        Placeholder for hepatic blood flow handling.
        """
        # A full model would track toxin clearance and nutrient processing.
        return


@dataclass
class HepaticSystem(Updatable):
    """
    Collection of hepatic functions, primarily the liver.
    """
    liver: Liver
    blood_glucose_mg_dl: float = 90.0

    def manage_energy(self, absorbed_kcal: float, body_energy_kcal: float) -> float:
        """
        Route absorbed calories to glycogen storage and release when low.
        Returns net calories available to general body stores.
        """
        available_for_body = 0.0

        # Store newly absorbed calories until glycogen capacity is reached.
        if absorbed_kcal > 0:
            capacity_remaining = max(0.0, self.liver.glycogen_capacity_kcal - self.liver.glycogen_store_kcal)
            stored = min(absorbed_kcal, capacity_remaining)
            self.liver.glycogen_store_kcal += stored
            available_for_body += absorbed_kcal - stored

        # Release glycogen if systemic energy is low.
        if body_energy_kcal < 1500.0 and self.liver.glycogen_store_kcal > 0:
            release = min(50.0, self.liver.glycogen_store_kcal)
            self.liver.glycogen_store_kcal -= release
            available_for_body += release

        # Update a coarse blood glucose estimate from glycogen status.
        fill_ratio = 0.0 if self.liver.glycogen_capacity_kcal <= 0 else self.liver.glycogen_store_kcal / self.liver.glycogen_capacity_kcal
        self.blood_glucose_mg_dl = 70.0 + (fill_ratio * 40.0)

        return available_for_body

    def step(self, ctx: TimeStepContext) -> None:
        """
        Placeholder for hepatic time-dependent processes.
        """
        self.liver.step(ctx)

    def energy_demand_watts(self) -> float:
        """
        Metabolic demand of hepatic tissue.
        """
        return self.liver.energy_demand_watts()

# ================================================================
# DIGESTIVE + METABOLIC SYSTEM
# ================================================================

@dataclass
class DigestiveSystem(Updatable):
    """
    Scaffold for processing food into energy.
    """
    stomach_content_kcal: float = 0.0
    intestine_content_kcal: float = 0.0
    absorption_rate_kcal_per_s: float = 5.0

    def ingest(self, calories: float) -> None:
        """
        Add food energy to the stomach.
        """
        self.stomach_content_kcal += calories

    def step(self, ctx: TimeStepContext) -> None:
        """
        Move calories through digestive tract and into bloodstream.
        """
        # Simple two-stage model: stomach → intestine → blood
        transfer_rate = self.absorption_rate_kcal_per_s * ctx.dt_seconds

        # Move from stomach to intestine
        stomach_transfer = min(self.stomach_content_kcal, transfer_rate)
        self.stomach_content_kcal -= stomach_transfer
        self.intestine_content_kcal += stomach_transfer

        # Absorb from intestine to "body" (handled externally)
        intestine_transfer = min(self.intestine_content_kcal, transfer_rate)
        self.intestine_content_kcal -= intestine_transfer

        # In a detailed model, this would increase blood glucose or fat stores.
        # Here we just expose the absorbed amount.
        self._last_absorbed_kcal = intestine_transfer  # type: ignore

    def pop_last_absorbed(self) -> float:
        """
        Retrieve the last absorbed kcal, reset to zero.
        """
        absorbed = getattr(self, "_last_absorbed_kcal", 0.0)
        self._last_absorbed_kcal = 0.0  # type: ignore
        return absorbed


# ================================================================
# RENAL SYSTEM
# ================================================================

@dataclass
class Kidney(MetabolicOrgan, CirculationSink):
    """
    Simple kidney scaffold for filtration and fluid balance.
    """
    filtration_rate_ml_per_min: float = 90.0
    hydration_contribution_pct: float = 0.0

    def receive_flow(self, volume_ml: float, oxygen_fraction: float) -> None:
        """
        Receive renal blood flow; placeholder for solute handling.
        """
        # A full model would calculate urine formation and electrolyte changes.
        return


@dataclass
class RenalSystem(Updatable):
    """
    Coarse renal model coordinating kidneys and hydration tracking.
    """
    kidneys: List[Kidney]
    hydration_level_pct: float = 100.0

    def _estimate_urine_output(self, ctx: TimeStepContext) -> float:
        """
        Estimate urine production from total filtration.
        """
        total_filtration_ml_per_s = sum(k.filtration_rate_ml_per_min for k in self.kidneys) / 60.0
        return total_filtration_ml_per_s * ctx.dt_seconds * 0.2  # assume 20% becomes urine

    def step(self, ctx: TimeStepContext) -> None:
        """
        Update kidney processes and hydration estimate.
        """
        urine_output_ml = self._estimate_urine_output(ctx)

        # Adjust hydration level heuristically.
        hydration_change = -urine_output_ml / 3000.0 * 100.0  # relative to a nominal 3L
        self.hydration_level_pct = max(0.0, min(120.0, self.hydration_level_pct + hydration_change))

    def energy_demand_watts(self) -> float:
        """
        Sum of metabolic demand from both kidneys.
        """
        return sum(k.energy_demand_watts() for k in self.kidneys)


# ================================================================
# IMMUNE SYSTEM
# ================================================================

@dataclass
class ImmuneSystem(Updatable):
    """
    Minimal immune activity tracker.
    """
    activation_level: float = 0.1  # 0–1
    pathogen_load: float = 0.0
    basal_metabolic_rate_watts: float = 5.0

    def challenge(self, load: float) -> None:
        """
        Introduce a pathogen load that the immune system must fight.
        """
        self.pathogen_load += max(0.0, load)
        self.activation_level = min(1.0, self.activation_level + load / 1000.0)

    def step(self, ctx: TimeStepContext) -> None:
        """
        Reduce pathogen load proportional to activation.
        """
        clearance_rate = self.activation_level * 10.0 * ctx.dt_seconds
        self.pathogen_load = max(0.0, self.pathogen_load - clearance_rate)

        # Activation slowly relaxes toward baseline when load is low.
        if self.pathogen_load <= 0:
            self.activation_level = max(0.1, self.activation_level - 0.01 * ctx.dt_seconds)

    def energy_demand_watts(self) -> float:
        """
        Metabolic demand increases with immune activation.
        """
        return self.basal_metabolic_rate_watts * (1.0 + self.activation_level)

# ================================================================
# HIGH-LEVEL BODY OBJECT
# ================================================================

@dataclass
class Body(Updatable):
    """
    High-level human body scaffold tying all systems together.
    """
    sex: Sex
    mass_kg: float
    height_m: float

    vital_signs: VitalSigns

    circulatory: CirculatorySystem
    respiratory: RespiratorySystem
    nervous: NervousSystem
    endocrine: EndocrineSystem
    hepatic: HepaticSystem
    renal: RenalSystem
    musculoskeletal: MusculoskeletalSystem
    digestive: DigestiveSystem
    immune: ImmuneSystem

    energy_stores_kcal: float = 2000.0

    def step(self, ctx: TimeStepContext) -> None:
        """
        Advance all body systems by one time step.
        """
        # Step sub-systems
        self.nervous.step(ctx)
        self.endocrine.step(ctx)
        self.musculoskeletal.step(ctx)
        self.respiratory.step(ctx)
        self.circulatory.step(ctx)
        self.digestive.step(ctx)
        self.hepatic.step(ctx)
        self.renal.step(ctx)
        self.immune.step(ctx)

        # Integrate metabolic flows
        self._update_energy_balance(ctx)
        self._update_vital_signs(ctx)

    def _update_energy_balance(self, ctx: TimeStepContext) -> None:
        """
        Integrate energy consumption and intake.
        """
        # Collect energy demand from organs that implement EnergyConsumer
        total_watts = 0.0

        # Heart
        total_watts += self.circulatory.heart.energy_demand_watts()

        # Lungs
        total_watts += self.respiratory.lungs.energy_demand_watts()

        # Muscles
        for m in self.musculoskeletal.muscles.values():
            total_watts += m.energy_demand_watts()

        # Brain and peripheral nerves
        total_watts += self.nervous.energy_demand_watts()

        # Endocrine regulation
        total_watts += self.endocrine.energy_demand_watts()

        # Hepatic processes
        total_watts += self.hepatic.energy_demand_watts()

        # Renal filtration
        total_watts += self.renal.energy_demand_watts()

        # Immune activity
        total_watts += self.immune.energy_demand_watts()

        # Convert Watts to kcal/s (1 Watt = 1 Joule/s; 1 kcal ≈ 4184 Joules)
        kcal_per_s = total_watts / 4184.0
        kcal_used = kcal_per_s * ctx.dt_seconds

        # Subtract from energy stores
        self.energy_stores_kcal = max(0.0, self.energy_stores_kcal - kcal_used)

        # Add calories absorbed from digestion
        absorbed_kcal = self.digestive.pop_last_absorbed()
        absorbed_for_body = self.hepatic.manage_energy(absorbed_kcal, self.energy_stores_kcal)
        self.energy_stores_kcal += absorbed_for_body

    def _update_vital_signs(self, ctx: TimeStepContext) -> None:
        """
        Update high-level vital signs based on system states.
        """
        # Example: set heart rate from heart model
        self.vital_signs.heart_rate_bpm = self.circulatory.heart.heart_rate_bpm

        # Example placeholder: vary temperature slightly with energy usage
        baseline_temp = 37.0
        # Simple heuristic: more energy → slightly higher temperature
        delta_temp = (self.energy_stores_kcal - 2000.0) / 10000.0
        self.vital_signs.body_temperature_c = baseline_temp + delta_temp

        # Track coarse blood glucose and hydration
        self.vital_signs.blood_glucose_mg_dl = self.hepatic.blood_glucose_mg_dl
        self.vital_signs.hydration_pct = self.renal.hydration_level_pct


# ================================================================
# FACTORY FUNCTION FOR A DEFAULT BODY
# ================================================================

def create_default_body(sex: Sex = Sex.MALE) -> Body:
    """
    Construct a default human body scaffold with coarse parameters.
    """
    # Create vital signs
    vital = VitalSigns()

    # Create major organs
    heart = Heart(
        name="Heart",
        mass_kg=0.3,
        basal_metabolic_rate_watts=15.0,
    )

    lungs = Lung(
        name="Lungs",
        mass_kg=1.0,
        basal_metabolic_rate_watts=10.0,
    )

    liver = Liver(
        name="Liver",
        mass_kg=1.5,
        basal_metabolic_rate_watts=18.0,
        glycogen_store_kcal=350.0,
    )

    # Create blood compartments
    arterial = BloodCompartment(volume_ml=2500.0, oxygen_fraction=0.98)
    venous = BloodCompartment(volume_ml=2500.0, oxygen_fraction=0.75)

    # Create circulatory system
    circulatory = CirculatorySystem(
        heart=heart,
        arterial=arterial,
        venous=venous,
    )

    # Create respiratory system
    respiratory = RespiratorySystem(lungs=lungs)

    # Create nervous system
    brain_network = NeuralNetwork()
    nervous = NervousSystem(central=brain_network)

    # Create endocrine system with a couple of baseline hormones
    endocrine = EndocrineSystem()
    endocrine.release("insulin", amount_ng_ml=5.0, half_life_s=240.0)
    endocrine.release("adrenaline", amount_ng_ml=0.5, half_life_s=120.0)

    # Create hepatic system
    hepatic = HepaticSystem(liver=liver)

    # Create renal system
    kidney_left = Kidney(
        name="Left Kidney",
        mass_kg=0.15,
        basal_metabolic_rate_watts=6.0,
        filtration_rate_ml_per_min=95.0,
    )
    kidney_right = Kidney(
        name="Right Kidney",
        mass_kg=0.15,
        basal_metabolic_rate_watts=6.0,
        filtration_rate_ml_per_min=95.0,
    )
    renal = RenalSystem(kidneys=[kidney_left, kidney_right])

    # Create musculoskeletal system (very minimal)
    femur = Bone(name="Femur", length_m=0.45, mass_kg=4.0)
    tibia = Bone(name="Tibia", length_m=0.43, mass_kg=3.0)
    knee = Joint(name="Knee", bone_a=femur, bone_b=tibia, min_angle_deg=0.0, max_angle_deg=135.0)

    quad = Muscle(
        name="Quadriceps",
        mass_kg=1.5,
        basal_metabolic_rate_watts=20.0,
    )

    muscles = {"Quadriceps": quad}
    bones = {"Femur": femur, "Tibia": tibia}
    joints = {"Knee": knee}

    musculoskeletal = MusculoskeletalSystem(
        bones=bones,
        joints=joints,
        muscles=muscles,
    )

    # Allow nervous system to control muscle
    nervous.register_target("Quadriceps", quad)

    # Digestive system
    digestive = DigestiveSystem()

    # Immune system
    immune = ImmuneSystem()

    # Register tissues as circulation sinks (e.g., muscles, lungs)
    circulatory.register_sink(lungs)
    for muscle in muscles.values():
        circulatory.register_sink(muscle)
    circulatory.register_sink(liver)
    circulatory.register_sink(kidney_left)
    circulatory.register_sink(kidney_right)

    # Create body
    body = Body(
        sex=sex,
        mass_kg=75.0,
        height_m=1.8,
        vital_signs=vital,
        circulatory=circulatory,
        respiratory=respiratory,
        nervous=nervous,
        endocrine=endocrine,
        hepatic=hepatic,
        renal=renal,
        musculoskeletal=musculoskeletal,
        digestive=digestive,
        immune=immune,
    )

    return body


# ================================================================
# MINIMAL EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":
    # Create a default body
    body = create_default_body()

    # Create a time step context (e.g., 0.1 seconds per step)
    ctx = SimpleTimeStepContext(dt_seconds=0.1)

    # Example: simulate 10 seconds of "time"
    total_time = 10.0
    steps = int(total_time / ctx.dt_seconds)

    # Example: ingest some food
    body.digestive.ingest(500.0)  # 500 kcal

    for i in range(steps):
        # Example: nervous system contracts quadriceps periodically
        if i % 5 == 0:
            body.nervous.send_signal("Quadriceps", "contract", magnitude=0.1)

        # Step the whole body
        body.step(ctx)

    # At this point, body.vital_signs and body.energy_stores_kcal hold a
    # coarse representation of the simulated state.

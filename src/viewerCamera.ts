import { addScaled, cross, dot, lookAt, normalize, perspective, subtract, type Mat4, type Vec3 } from './mat4';

/** A viewpoint to jump to: typically where a training photo was taken. */
export interface CameraPose {
  position: Vec3;
  /** The direction the camera looks, unit length. */
  forward: Vec3;
  fovY: number;
}

export interface ViewerCameraOptions {
  /** The point to orbit. */
  target?: Vec3;
  /** The scene's up direction. Trained scenes often aren't +y. */
  up?: Vec3;
  /** Where to start: the camera looks from here at the target. */
  eye?: Vec3;
  /** Roughly the scene's size. Sets the walking speed and the zoom limits. */
  radius?: number;
  /** Viewpoints for [ and ] to step through. */
  poses?: CameraPose[];
}

const DEFAULT_FOV = Math.PI / 4;
const TURN_SPEED = 0.005; // radians per pixel dragged
const SPIN_SPEED = 0.15; // radians per second, while idle

// The viewer's camera (M1.6), in two modes:
// - orbit circles a target point: drag to turn around it, scroll to zoom;
// - walk moves freely: WASD or the arrow keys, Q and E for down and up, Shift to go
//   faster; drag to look around, scroll to move forward and back.
// Moving switches to walk mode and O goes back to orbiting. [ and ] jump between
// the training cameras; R resets. It spins slowly on its own until the first input.
export class ViewerCamera {
  mode: 'orbit' | 'walk' = 'orbit';
  fovY = DEFAULT_FOV;
  private readonly up: Vec3;
  // Two horizontal directions, at right angles to up and to each other.
  private readonly east: Vec3;
  private readonly north: Vec3;
  private readonly radius: number;
  private readonly poses: CameraPose[];
  private readonly home: { target: Vec3; distance: number; yaw: number; pitch: number };
  private target: Vec3;
  private distance: number;
  private position: Vec3 = [0, 0, 0]; // walk mode's eye
  // Where the camera looks: a turn around up (0 is north) and an angle above the horizon.
  private yaw = 0;
  private pitch = 0;
  private poseIndex = -1;
  private spinning = true;
  private readonly keys = new Set<string>();

  constructor(element: HTMLElement, options: ViewerCameraOptions = {}) {
    this.up = normalize(options.up ?? [0, 1, 0]);
    const reference: Vec3 = Math.abs(this.up[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    this.east = normalize(cross(reference, this.up));
    this.north = cross(this.up, this.east);
    this.radius = options.radius ?? 1;
    this.poses = options.poses ?? [];
    this.target = options.target ?? [0, 0, 0];
    const toTarget = subtract(this.target, options.eye ?? addScaled(this.target, this.north, -3 * this.radius));
    this.distance = Math.hypot(...toTarget);
    this.lookAlong(normalize(toTarget));
    this.home = { target: this.target, distance: this.distance, yaw: this.yaw, pitch: this.pitch };
    this.listen(element);
  }

  /** What the camera is doing, for the status line. */
  get label(): string {
    if (this.mode === 'orbit') return 'orbit';
    return this.poseIndex >= 0 ? `photo ${this.poseIndex + 1} of ${this.poses.length}` : 'walk';
  }

  /** Which pose the camera is exactly at, or −1. Turning, zooming or moving leaves it. */
  get photoIndex(): number {
    return this.poseIndex;
  }

  /** Jumps to pose `index`. */
  goToPhoto(index: number): void {
    this.jumpTo(this.poses[index]);
    this.poseIndex = index;
  }

  /** Advances walking and the idle spin by `seconds`. */
  update(seconds: number): void {
    if (this.spinning && this.mode === 'orbit') this.yaw += seconds * SPIN_SPEED;
    const forward = this.direction();
    const right = normalize(cross(forward, this.up));
    const pressed = (...codes: string[]) => codes.some((code) => this.keys.has(code));
    let move: Vec3 = [0, 0, 0];
    if (pressed('KeyW', 'ArrowUp')) move = addScaled(move, forward, 1);
    if (pressed('KeyS', 'ArrowDown')) move = addScaled(move, forward, -1);
    if (pressed('KeyD', 'ArrowRight')) move = addScaled(move, right, 1);
    if (pressed('KeyA', 'ArrowLeft')) move = addScaled(move, right, -1);
    if (pressed('KeyE')) move = addScaled(move, this.up, 1);
    if (pressed('KeyQ')) move = addScaled(move, this.up, -1);
    if (move.some((value) => value !== 0)) {
      this.startWalking();
      const speed = 0.5 * this.radius * (pressed('ShiftLeft', 'ShiftRight') ? 3 : 1);
      this.position = addScaled(this.position, move, speed * seconds);
    }
  }

  matrices(aspect: number): { view: Mat4; proj: Mat4 } {
    const eye = this.mode === 'orbit' ? addScaled(this.target, this.direction(), -this.distance) : this.position;
    return {
      view: lookAt(eye, addScaled(eye, this.direction(), 1), this.up),
      proj: perspective(this.fovY, aspect, 0.001 * this.radius, 1000 * this.radius),
    };
  }

  /** Looks from exactly where `pose` looked from. Any roll around the view axis is dropped. */
  jumpTo(pose: CameraPose): void {
    this.mode = 'walk';
    this.spinning = false;
    this.position = pose.position;
    this.fovY = pose.fovY;
    this.lookAlong(pose.forward);
  }

  private listen(element: HTMLElement): void {
    element.addEventListener('pointerdown', (event) => {
      this.spinning = false;
      element.setPointerCapture(event.pointerId);
    });
    element.addEventListener('pointermove', (event) => {
      if (!element.hasPointerCapture(event.pointerId)) return;
      // Either way, the scene follows the pointer: orbiting turns the camera the
      // opposite way, looking around turns it the same way.
      const sign = this.mode === 'orbit' ? 1 : -1;
      if (event.movementX || event.movementY) this.poseIndex = -1;
      this.yaw += sign * event.movementX * TURN_SPEED;
      this.pitch = clamp(this.pitch - sign * event.movementY * TURN_SPEED, -1.5, 1.5);
    });
    element.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        this.spinning = false;
        this.poseIndex = -1;
        if (this.mode === 'orbit') {
          this.distance = clamp(this.distance * Math.exp(event.deltaY * 0.001), 0.05 * this.radius, 20 * this.radius);
        } else {
          this.position = addScaled(this.position, this.direction(), -0.002 * event.deltaY * this.radius);
        }
      },
      { passive: false },
    );
    // event.code names physical keys, so WASD works whatever the keyboard layout.
    window.addEventListener('keydown', (event) => {
      this.keys.add(event.code);
      this.spinning = false;
      if (event.code === 'BracketRight') this.stepPose(1);
      if (event.code === 'BracketLeft') this.stepPose(-1);
      if (event.code === 'KeyO') this.orbit();
      if (event.code === 'KeyR') this.reset();
    });
    window.addEventListener('keyup', (event) => this.keys.delete(event.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  private direction(): Vec3 {
    const horizontal = Math.cos(this.pitch);
    const e = horizontal * Math.sin(this.yaw);
    const n = horizontal * Math.cos(this.yaw);
    const u = Math.sin(this.pitch);
    return [
      e * this.east[0] + n * this.north[0] + u * this.up[0],
      e * this.east[1] + n * this.north[1] + u * this.up[1],
      e * this.east[2] + n * this.north[2] + u * this.up[2],
    ];
  }

  private lookAlong(forward: Vec3): void {
    this.pitch = Math.asin(clamp(dot(forward, this.up), -1, 1));
    this.yaw = Math.atan2(dot(forward, this.east), dot(forward, this.north));
  }

  private startWalking(): void {
    if (this.mode === 'orbit') this.position = addScaled(this.target, this.direction(), -this.distance);
    this.mode = 'walk';
    this.spinning = false;
    this.poseIndex = -1;
  }

  // Back to orbiting, around the point straight ahead at the current distance.
  private orbit(): void {
    if (this.mode === 'walk') this.target = addScaled(this.position, this.direction(), this.distance);
    this.mode = 'orbit';
    this.fovY = DEFAULT_FOV;
    this.poseIndex = -1;
  }

  private reset(): void {
    this.mode = 'orbit';
    this.fovY = DEFAULT_FOV;
    this.poseIndex = -1;
    ({ target: this.target, distance: this.distance, yaw: this.yaw, pitch: this.pitch } = this.home);
    this.spinning = true;
  }

  private stepPose(delta: number): void {
    const count = this.poses.length;
    if (!count) return;
    this.goToPhoto(this.poseIndex < 0 ? (delta > 0 ? 0 : count - 1) : (this.poseIndex + delta + count) % count);
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

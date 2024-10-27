<template>
  <UContainer class="max-w-md mx-auto h-full overflow-y-scroll">
    <h1 class="text-4xl font-bold w-fit mx-auto">⚡ Turbo Trash</h1>

    <UDivider class="py-6" />

    <ClientOnly fallback-tag="div">
      <template #fallback>
        <div class="flex flex-col justify-center items-center">
          <UIcon class="text-4xl mx-auto" name="svg-spinners:ring-resize" />

          <span class="py-4"> Baking the model into delicious cookies... </span>
        </div>
      </template>

      <div class="h-fit flex flex-col gap-6">
        <div v-if="finalized">
          <USelect
            v-model="videoDevice"
            :options="cameraOptions"
            option-attribute="label"
          />
          <video
            ref="video"
            muted
            controls
            autoplay
            class="h-fit w-full rounded-lg mt-4"
          />
        </div>
        <USkeleton class="h-60 w-full" v-else />

        <div class="max-h-56 overflow-y-hidden" v-if="finalized">
          <div class="flex justify-center flex-col items-center py-6 gap-4">
            <span class="text-3xl">{{ categories[type] }}</span>

            <TransitionFade group tag="ul">
              <li>It may be...</li>
              <li
                v-for="item in prediction"
                :key="item.label"
                class="transition-all duration-100"
                :class="{
                  'opacity-0': item.prob < threshold,
                }"
              >
                {{ item.label }}
              </li>
            </TransitionFade>
          </div>
        </div>
        <div class="h-fit" v-else>
          <div class="flex justify-center flex-col items-center py-6 gap-4">
            <USkeleton class="h-8 w-64" />
            <ul class="flex flex-col gap-2">
              <li>
                <USkeleton class="h-4 w-32" />
              </li>
              <li>
                <USkeleton class="h-4 w-32" />
              </li>
              <li>
                <USkeleton class="h-4 w-32" />
              </li>
              <li>
                <USkeleton class="h-4 w-32" />
              </li>
            </ul>
          </div>
        </div>
      </div>
    </ClientOnly>

    <UDivider class="py-6" />

    <UButton
      block
      :to="`/map?filter=${type}&lnglat=${longitude},${latitude}`"
      color="primary"
      class="text-xl"
      v-if="finalized"
    >
      🌍 Show me its destination!
    </UButton>

    <canvas ref="canvas" class="h-0 w-0" v-show="false" />
  </UContainer>
</template>

<script setup lang="ts">
// Height and width of the input
const CW = 256;
const CH = 256;
const GAMMA = 0.5;
const categories = ["❌ Non-recyclable", "✅ Recyclable"];
const labels = [
  { label: "🔋 Battery", type: 0 },
  { label: "🌿 Biological", type: 0 },
  { label: "🥛 Brown glass", type: 1 },
  { label: "📦 Cardboard", type: 1 },
  { label: "👗 Clothes", type: 0 },
  { label: "🍾 Green glass", type: 1 },
  { label: "🛠 Metal", type: 1 },
  { label: "📄 Paper", type: 1 },
  { label: "🥤 Plastic", type: 1 },
  { label: "👟 Shoes", type: 0 },
  { label: "🗑 Trash", type: 0 },
  { label: "🍶 White glass", type: 1 },
];
const threshold = 1 / labels.length;

const pause = ref(false);

// For location
const { coords } = useGeolocation();
const latitude = computed(() =>
  coords.value?.latitude !== Number.POSITIVE_INFINITY
    ? coords.value?.latitude
    : 35.227085
);
const longitude = computed(() =>
  coords.value?.longitude !== Number.POSITIVE_INFINITY
    ? coords.value?.longitude
    : -80.843124
);

const videoDevice = ref<string>();
const { videoInputs: cameras } = useDevicesList({
  requestPermissions: true,
  onUpdated: () => {
    if (!cameras.value.find((i) => i.deviceId === videoDevice.value))
      videoDevice.value = cameras.value[0]?.deviceId;
  },
});
const video = ref<HTMLVideoElement>();
const cameraOptions = computed(() => {
  return cameras.value.map((camera) => ({
    label: camera.label,
    value: camera.deviceId,
  }));
});
const { stream, restart } = useUserMedia({
  constraints: { video: { deviceId: videoDevice }, audio: false },
  enabled: true,
  autoSwitch: true,
});
watchEffect(() => {
  if (video.value) video.value.srcObject = stream.value!;
});
watch(videoDevice, () => {
  pause.value = true;
  restart();
  pause.value = false;
});

const canvas = ref<HTMLCanvasElement>();
const ctx = ref<CanvasRenderingContext2D>();
watchOnce(canvas, () => {
  if (!canvas.value) return;
  canvas.value.width = CW;
  canvas.value.height = CH;
  const _ctx = canvas.value.getContext("2d");
  if (!_ctx) {
    console.error("Failed to get 2D context");
    return;
  }
  ctx.value = _ctx;
});

// Moving average
const probs = ref(Array(labels.length).fill(threshold) as number[]);
const prediction = computed(() => {
  // Top 3
  return labels
    .map((l, i) => ({ prob: probs.value[i], label: l.label }))
    .sort((a, b) => b.prob - a.prob);
});
const type = computed(() => {
  const _prob = [0, 0];
  for (let i = 0; i < labels.length; i++) {
    _prob[labels[i].type] += probs.value[i];
  }
  return _prob[0] > _prob[1] ? 0 : 1;
});

const finalized = ref(false);

import init, { start, GarbageClassification } from "classification-model";
onMounted(async () => {
  // Async import

  await init();
  await start();

  const model = new GarbageClassification();

  finalized.value = true;

  const update = async () => {
    if (pause.value) return;
    const vw = stream.value?.getVideoTracks()[0].getSettings().width;
    const vh = stream.value?.getVideoTracks()[0].getSettings().height;
    if (!vw || !vh || !canvas.value || !video.value || !ctx.value) return;

    let pw = 0;
    let ph = 0;
    // Calculate the aspect ratio
    if (vw / vh > CW / CH) {
      pw = (vw - vh) / 2;
    } else {
      ph = (vh - vw) / 2;
    }

    ctx.value.drawImage(
      video.value,
      // Padding
      pw,
      ph,
      // Cropped width and height
      vw - 2 * pw,
      vh - 2 * ph,
      // Destination
      0,
      0,
      // Destination width and height
      CW,
      CH
    );
    const result = await model.inference(
      new Float32Array(ctx.value.getImageData(0, 0, CW, CH).data)
    );
    // Update the moving average
    probs.value = probs.value.map(
      (p, i) => p * GAMMA + result[i] * (1 - GAMMA)
    );
  };

  const loop = async () => {
    await new Promise((resolve) => setTimeout(resolve, 100)); // Throttle
    await update();
    requestAnimationFrame(loop);
  };

  await loop(); // Never-ending loop
});
</script>

<template>
  <div class="h-full w-full">
    <ClientOnly>
      <template #fallback>
        <USkeleton class="h-full w-full" />
      </template>
      <MglMap
        :map-style="style"
        :center="lnglat"
        :zoom="zoom"
        width="100%"
        height="100%"
      >
        <MglFullscreenControl />
        <MglNavigationControl />
        <MglGeolocateControl
          :track-user-location="true"
          :position-options="{ enableHighAccuracy: true }"
          :fit-bounds-options="{ maxZoom: 18 }"
          ref="trackLocationButton"
        />
        <MglCustomControl position="top-right">
          <button
            class="maplibregl-ctrl-icon"
            @click="() => (showAll = !showAll)"
          >
            <UIcon
              v-if="showAll"
              name="mdi:filter-remove"
              class="text-gray-500 text-lg mt-1"
            />
            <UIcon
              v-else
              name="mdi:filter"
              class="text-gray-900 text-lg mt-1"
            />
          </button>
        </MglCustomControl>

        <!-- Trash marker -->
        <MglMarker
          v-for="item in coords.trash"
          :coordinates="item"
          v-if="filter != 1 || showAll"
        >
          <template v-slot:marker>
            <div class="h-0 w-0 overflow-visible">
              <div
                class="h-8 w-8 bg-white bg-opacity-50 border border-gray-900 border-opacity-50 rounded-full rounded-br-none rotate-45 -translate-y-8 -translate-x-4"
              >
                <UIcon
                  name="mdi:trash-can"
                  class="text-gray-500 text-3xl -rotate-45"
                />
              </div>
            </div>
          </template>
        </MglMarker>
        <!-- Recycle marker -->
        <MglMarker
          v-for="item in coords.recycle"
          :coordinates="item"
          v-if="filter != 0 || showAll"
        >
          <template v-slot:marker>
            <div class="h-0 w-0 overflow-visible">
              <div
                class="h-8 w-8 bg-white bg-opacity-50 border border-gray-900 border-opacity-50 rounded-full rounded-br-none rotate-45 -translate-y-8 -translate-x-4"
              >
                <UIcon
                  name="mdi:recycle"
                  class="text-lime-500 text-3xl -rotate-45"
                />
              </div>
            </div>
          </template>
        </MglMarker>
      </MglMap>
    </ClientOnly>
  </div>
</template>

<script setup lang="ts">
const { coords: live } = useGeolocation();
const style =
  "https://api.maptiler.com/maps/streets/style.json?key=cQX2iET1gmOW38bedbUh";
const zoom = 18;

const route = useRoute();
const filter = ref(
  // Default to no filter
  Number((route.query.filter as string) || "-1")
);
const showAll = ref(false);
const lnglat = ref(
  // Default to center of Charlotte, NC
  ((route.query.lnglat as string) || "-80.843124,35.227085")
    .split(",")
    .map(Number)
);

import coords from "~/assets/coords.json";
</script>

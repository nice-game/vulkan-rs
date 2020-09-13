use crate::{
	instance::Instance,
	surface::{PresentMode, Surface, SurfaceCapabilities, SurfaceFormat},
};
use ash::{version::InstanceV1_0, vk};
use std::sync::Arc;

#[derive(Clone)]
pub struct PhysicalDevice {
	instance: Arc<Instance>,
	pub vk: vk::PhysicalDevice,
}
impl PhysicalDevice {
	pub fn enumerate<'a>(instance: &'a Arc<Instance>) -> impl Iterator<Item = Arc<Self>> + 'a {
		unsafe { instance.vk.enumerate_physical_devices() }
			.unwrap()
			.into_iter()
			.map(move |vk| Arc::new(Self { instance: instance.clone(), vk }))
	}

	pub fn get_queue_family_properties<'a>(
		self: &'a Arc<PhysicalDevice>,
	) -> impl Iterator<Item = QueueFamilyProperties> + 'a {
		unsafe { self.instance.vk.get_physical_device_queue_family_properties(self.vk) }.into_iter().enumerate().map(
			move |(i, vk)| QueueFamilyProperties { family: unsafe { QueueFamily::from_vk(self.clone(), i as _) }, vk },
		)
	}

	pub fn get_surface_capabilities<T>(&self, surface: &Surface<T>) -> SurfaceCapabilities {
		unsafe { self.instance.khr_surface.get_physical_device_surface_capabilities(self.vk, surface.vk) }.unwrap()
	}

	pub fn get_surface_formats<T>(&self, surface: &Surface<T>) -> Vec<SurfaceFormat> {
		unsafe { self.instance.khr_surface.get_physical_device_surface_formats(self.vk, surface.vk) }.unwrap()
	}

	pub fn get_surface_present_modes<T>(&self, surface: &Surface<T>) -> Vec<PresentMode> {
		unsafe { self.instance.khr_surface.get_physical_device_surface_present_modes(self.vk, surface.vk) }.unwrap()
	}

	pub fn get_surface_support<T>(&self, qfam: &QueueFamily, surface: &Surface<T>) -> bool {
		unsafe { self.instance.khr_surface.get_physical_device_surface_support(self.vk, qfam.idx, surface.vk) }.unwrap()
	}

	pub fn instance(&self) -> &Arc<Instance> {
		&self.instance
	}
}
impl PartialEq for PhysicalDevice {
	fn eq(&self, other: &PhysicalDevice) -> bool {
		self.vk == other.vk
	}
}
impl Eq for PhysicalDevice {}

pub struct QueueFamilyProperties {
	family: QueueFamily,
	vk: vk::QueueFamilyProperties,
}
impl QueueFamilyProperties {
	pub fn family(self) -> QueueFamily {
		self.family
	}

	pub fn queue_flags(&self) -> QueueFlags {
		QueueFlags { vk: self.vk.queue_flags }
	}
}

#[derive(Clone, PartialEq, Eq)]
pub struct QueueFamily {
	physical_device: Arc<PhysicalDevice>,
	pub idx: u32,
}
impl QueueFamily {
	pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
		&self.physical_device
	}

	pub(crate) unsafe fn from_vk(physical_device: Arc<PhysicalDevice>, idx: u32) -> Self {
		Self { physical_device, idx }
	}
}

#[derive(Clone, Copy)]
pub struct QueueFlags {
	vk: vk::QueueFlags,
}
impl QueueFlags {
	pub fn graphics(self) -> bool {
		self.vk.contains(vk::QueueFlags::GRAPHICS)
	}
}

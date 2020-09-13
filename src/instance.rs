use crate::Vulkan;
use ash::{
	extensions::khr,
	version::{EntryV1_0, InstanceV1_0},
	vk, Instance as VkInstance,
};
use std::{
	collections::HashSet,
	ffi::{CStr, CString},
	sync::Arc,
};

pub struct Instance {
	_vulkan: Arc<Vulkan>,
	pub vk: VkInstance,
	pub khr_surface: khr::Surface,
	#[cfg(windows)]
	pub khr_win32_surface: khr::Win32Surface,
	#[cfg(unix)]
	pub khr_xlib_surface: khr::XlibSurface,
	#[cfg(unix)]
	pub khr_wayland_surface: khr::WaylandSurface,
}
impl Instance {
	pub fn new(vulkan: Arc<Vulkan>, application_name: &str, application_version: Version) -> Arc<Self> {
		let application_name = CString::new(application_name).unwrap();

		let app_info = vk::ApplicationInfo::builder()
			.application_name(&application_name)
			.application_version(application_version.vk);

		let mut exts = vec![b"VK_KHR_surface\0".as_ptr() as _];
		#[cfg(windows)]
		exts.push(b"VK_KHR_win32_surface\0".as_ptr() as _);
		#[cfg(unix)]
		exts.push(b"VK_KHR_xlib_surface\0".as_ptr() as _);

		#[allow(unused_mut)]
		let mut layers_pref = HashSet::new();
		// #[cfg(debug_assertions)]
		// layers_pref.insert(CStr::from_bytes_with_nul(b"VK_LAYER_LUNARG_monitor\0").unwrap());
		let layers = vulkan.vk.enumerate_instance_layer_properties().unwrap();
		let layers = layers
			.iter()
			.map(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) })
			.collect::<HashSet<_>>()
			.intersection(&layers_pref)
			.map(|ext| ext.as_ptr())
			.collect::<Vec<_>>();

		let ci = vk::InstanceCreateInfo::builder()
			.application_info(&app_info)
			.enabled_layer_names(&layers)
			.enabled_extension_names(&exts);
		let vk = unsafe { vulkan.vk.create_instance(&ci, None) }.unwrap();
		let khr_surface = khr::Surface::new(&vulkan.vk, &vk);
		#[cfg(windows)]
		let khr_win32_surface = khr::Win32Surface::new(&vulkan.vk, &vk);
		#[cfg(unix)]
		let khr_xlib_surface = khr::XlibSurface::new(&vulkan.vk, &vk);
		#[cfg(unix)]
		let khr_wayland_surface = khr::WaylandSurface::new(&vulkan.vk, &vk);

		Arc::new(Self {
			_vulkan: vulkan,
			vk,
			khr_surface,
			#[cfg(windows)]
			khr_win32_surface,
			#[cfg(unix)]
			khr_xlib_surface,
			#[cfg(unix)]
			khr_wayland_surface,
		})
	}
}
impl Drop for Instance {
	fn drop(&mut self) {
		unsafe {
			self.vk.destroy_instance(None);
		}
	}
}

pub struct Version {
	vk: u32,
}
impl Version {
	pub fn new(major: u32, minor: u32, patch: u32) -> Self {
		Self { vk: vk::make_version(major, minor, patch) }
	}
}

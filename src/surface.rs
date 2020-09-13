pub use ash::vk::{
	ColorSpaceKHR as ColorSpace, PresentModeKHR as PresentMode, SurfaceCapabilitiesKHR as SurfaceCapabilities,
	SurfaceFormatKHR as SurfaceFormat, SurfaceTransformFlagsKHR as SurfaceTransformFlags,
};

use crate::instance::Instance;
use ash::vk;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

pub struct Surface<T> {
	instance: Arc<Instance>,
	window: T,
	pub vk: vk::SurfaceKHR,
}
impl<T: HasRawWindowHandle> Surface<T> {
	pub fn new(instance: Arc<Instance>, window: T) -> Arc<Surface<T>> {
		let vk = match window.raw_window_handle() {
			#[cfg(windows)]
			RawWindowHandle::Windows(handle) => {
				let ci = vk::Win32SurfaceCreateInfoKHR::builder().hinstance(handle.hinstance).hwnd(handle.hwnd);
				unsafe { instance.khr_win32_surface.create_win32_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Xlib(handle) => {
				let ci = vk::XlibSurfaceCreateInfoKHR::builder().dpy(handle.display as _).window(handle.window);
				unsafe { instance.khr_xlib_surface.create_xlib_surface(&ci, None) }.unwrap()
			},
			#[cfg(unix)]
			RawWindowHandle::Wayland(handle) => {
				let ci = vk::WaylandSurfaceCreateInfoKHR::builder().display(handle.display).surface(handle.surface);
				unsafe { instance.khr_wayland_surface.create_wayland_surface(&ci, None) }.unwrap()
			},
			_ => unimplemented!(),
		};

		Arc::new(Self { instance, window, vk })
	}
}
impl<T> Surface<T> {
	pub fn window(&self) -> &T {
		&self.window
	}
}
impl<T> Drop for Surface<T> {
	fn drop(&mut self) {
		unsafe { self.instance.khr_surface.destroy_surface(self.vk, None) };
	}
}

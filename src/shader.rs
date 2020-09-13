pub use ash::vk::ShaderStageFlags;

use crate::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct ShaderModule {
	device: Arc<Device>,
	pub vk: vk::ShaderModule,
}
impl ShaderModule {
	pub unsafe fn new(device: Arc<Device>, code: &[u32]) -> Arc<ShaderModule> {
		let ci = vk::ShaderModuleCreateInfo::builder().code(code);
		let vk = device.vk.create_shader_module(&ci, None).unwrap();
		Arc::new(Self { device, vk })
	}
}
impl Drop for ShaderModule {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_shader_module(self.vk, None) };
	}
}

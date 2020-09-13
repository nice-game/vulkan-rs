pub mod buffer;
pub mod command;
pub mod descriptor;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod pipeline;
pub mod render_pass;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;

pub use ash::{
	vk::{Extent2D, Offset2D, Rect2D, Result as VkResult},
	LoadingError,
};

use crate::instance::Instance;
use ash::Entry;
use std::sync::Arc;

pub struct Vulkan {
	pub vk: Entry,
}
impl Vulkan {
	pub fn new() -> Result<Arc<Self>, LoadingError> {
		Ok(Arc::new(Self { vk: Entry::new()? }))
	}
}

pub use ash::vk::{
	ClearColorValue, Format, ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageType, ImageUsageFlags,
};

use crate::{
	buffer::Buffer,
	command::{CommandPool, ImageMemoryBarrier, PipelineStageFlags},
	device::{Device, Queue, SubmitFuture},
	render_pass::RenderPass,
};
use ash::{version::DeviceV1_0, vk};
use atomic::Atomic;
use nalgebra::Vector3;
use std::{
	iter::once,
	sync::{atomic::Ordering, Arc},
};
use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

pub struct Image {
	device: Arc<Device>,
	pub(crate) vk: vk::Image,
	alloc: Allocation,
	size: Vector3<u32>,
	layout: Atomic<ImageLayout>,
}
impl Image {
	pub fn init(
		device: Arc<Device>,
		image_type: ImageType,
		width: u32,
		height: u32,
		depth: u32,
		format: Format,
		usage: ImageUsageFlags,
	) -> ImageInit {
		let extent = vk::Extent3D::builder().width(width).height(height).depth(depth).build();
		let layout = ImageLayout::UNDEFINED;
		let ci = vk::ImageCreateInfo::builder()
			.image_type(image_type)
			.format(format)
			.extent(extent)
			.mip_levels(1)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.tiling(vk::ImageTiling::OPTIMAL)
			.usage(usage)
			.sharing_mode(vk::SharingMode::EXCLUSIVE)
			.initial_layout(layout);

		let usage = MemoryUsage::GpuOnly;
		let aci = AllocationCreateInfo { usage, ..Default::default() };

		let (vk, alloc, _) = device.allocator.create_image(&ci, &aci).unwrap();

		let size = Vector3::new(width, height, depth);

		let buf = Arc::new(Self { device, vk, alloc, size, layout: Atomic::new(layout) });
		ImageInit::new(buf)
	}

	pub fn size(&self) -> &Vector3<u32> {
		&self.size
	}

	pub fn layout(&self) -> ImageLayout {
		self.layout.load(Ordering::Relaxed)
	}

	pub fn set_layout(&self, layout: ImageLayout) {
		self.layout.store(layout, Ordering::Relaxed)
	}

	pub fn len(&self) -> u64 {
		self.size.x as u64 * self.size.y as u64 * self.size.z as u64
	}
}
impl ImageAbstract for Image {
	fn device(&self) -> &Arc<Device> {
		&self.device
	}

	fn vk(&self) -> vk::Image {
		self.vk
	}
}
impl Drop for Image {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_image(self.vk, None) };
		self.device.allocator.free_memory(&self.alloc).unwrap();
	}
}

pub struct ImageInit {
	pub(crate) img: Arc<Image>,
}
impl ImageInit {
	fn new(img: Arc<Image>) -> Self {
		Self { img }
	}
}
impl ImageInit {
	pub fn clear(
		self,
		queue: &Arc<Queue>,
		pool: &Arc<CommandPool>,
		color: ClearColorValue,
	) -> (Arc<Image>, SubmitFuture) {
		let cmd = pool
			.record(true, false)
			.pipeline_barrier(
				PipelineStageFlags::TOP_OF_PIPE,
				PipelineStageFlags::TRANSFER,
				once(ImageMemoryBarrier::new(self.img.clone(), ImageLayout::TRANSFER_DST_OPTIMAL)),
			)
			.clear_color_image(self.img.clone(), color)
			.pipeline_barrier(
				PipelineStageFlags::TRANSFER,
				PipelineStageFlags::FRAGMENT_SHADER | PipelineStageFlags::VERTEX_SHADER,
				once(ImageMemoryBarrier::new(self.img.clone(), ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
			)
			.build();
		let future = queue.submit(cmd);
		(self.img, future)
	}

	pub fn copy_from_buffer<T: Send + Sync + 'static>(
		self,
		queue: &Arc<Queue>,
		pool: &Arc<CommandPool>,
		buffer: Arc<Buffer<[T]>>,
	) -> (Arc<Image>, SubmitFuture) {
		let cmd = pool
			.record(true, false)
			.pipeline_barrier(
				PipelineStageFlags::TOP_OF_PIPE,
				PipelineStageFlags::TRANSFER,
				once(ImageMemoryBarrier::new(self.img.clone(), ImageLayout::TRANSFER_DST_OPTIMAL)),
			)
			.copy_buffer_to_image(buffer, self.img.clone(), &self.img.size)
			.pipeline_barrier(
				PipelineStageFlags::TRANSFER,
				PipelineStageFlags::FRAGMENT_SHADER | PipelineStageFlags::VERTEX_SHADER,
				once(ImageMemoryBarrier::new(self.img.clone(), ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
			)
			.build();
		let future = queue.submit(cmd);
		(self.img, future)
	}
}

pub struct Framebuffer {
	render_pass: Arc<RenderPass>,
	_attachments: Vec<Arc<ImageView>>,
	pub vk: vk::Framebuffer,
}
impl Framebuffer {
	pub fn new(
		device: Arc<Device>,
		render_pass: Arc<RenderPass>,
		attachments: Vec<Arc<ImageView>>,
		width: u32,
		height: u32,
	) -> Arc<Framebuffer> {
		let attachment_vks: Vec<_> = attachments.iter().map(|x| x.vk).collect();

		let ci = vk::FramebufferCreateInfo::builder()
			.render_pass(render_pass.vk)
			.attachments(&attachment_vks)
			.width(width)
			.height(height)
			.layers(1);
		let vk = unsafe { device.vk.create_framebuffer(&ci, None) }.unwrap();
		Arc::new(Self { render_pass, _attachments: attachments, vk })
	}
}
impl Drop for Framebuffer {
	fn drop(&mut self) {
		unsafe { self.render_pass.device().vk.destroy_framebuffer(self.vk, None) };
	}
}

pub struct ImageView {
	image: Arc<dyn ImageAbstract + Send + Sync>,
	pub vk: vk::ImageView,
}
impl ImageView {
	pub fn new(
		image: Arc<dyn ImageAbstract + Send + Sync>,
		format: Format,
		subresource_range: ImageSubresourceRange,
	) -> Arc<ImageView> {
		let ci = vk::ImageViewCreateInfo::builder()
			.image(image.vk())
			.view_type(vk::ImageViewType::TYPE_2D)
			.format(format)
			.subresource_range(subresource_range);
		let vk = unsafe { image.device().vk.create_image_view(&ci, None) }.unwrap();
		Arc::new(Self { image, vk })
	}
}
impl ImageAbstract for ImageView {
	fn device(&self) -> &Arc<Device> {
		self.image.device()
	}

	fn vk(&self) -> vk::Image {
		self.image.vk()
	}
}
impl Drop for ImageView {
	fn drop(&mut self) {
		unsafe { self.image.device().vk.destroy_image_view(self.vk, None) };
	}
}

pub struct Sampler {
	device: Arc<Device>,
	pub(crate) vk: vk::Sampler,
}
impl Sampler {
	pub fn new(device: Arc<Device>) -> Arc<Self> {
		let ci = vk::SamplerCreateInfo::builder()
			.mag_filter(vk::Filter::NEAREST)
			.min_filter(vk::Filter::NEAREST)
			.mipmap_mode(vk::SamplerMipmapMode::NEAREST)
			.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
		let vk = unsafe { device.vk.create_sampler(&ci, None) }.unwrap();
		Arc::new(Self { device, vk })
	}
}
impl Drop for Sampler {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_sampler(self.vk, None) };
	}
}

pub trait ImageAbstract {
	fn device(&self) -> &Arc<Device>;
	fn vk(&self) -> vk::Image;
}

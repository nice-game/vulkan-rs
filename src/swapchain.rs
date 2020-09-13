use crate::{
	device::Queue,
	image::ImageAbstract,
	physical_device::QueueFamily,
	sync::{GpuFuture, Semaphore, SemaphoreFuture, SubmitState},
};
pub use ash::vk::CompositeAlphaFlagsKHR as CompositeAlphaFlags;

use crate::{
	device::Device,
	image::Format,
	surface::{ColorSpace, PresentMode, Surface, SurfaceTransformFlags},
	Extent2D,
};
use ash::vk;
use std::sync::Arc;

trait SwapchainAbstract {
	fn device(&self) -> &Arc<Device>;
}

pub struct Swapchain<T> {
	device: Arc<Device>,
	surface: Arc<Surface<T>>,
	pub vk: vk::SwapchainKHR,
}
impl<T: Send + Sync + 'static> Swapchain<T> {
	pub fn new(
		device: Arc<Device>,
		surface: Arc<Surface<T>>,
		min_image_count: u32,
		image_format: Format,
		image_color_space: ColorSpace,
		image_extent: Extent2D,
		queue_families: impl IntoIterator<Item = QueueFamily>,
		pre_transform: SurfaceTransformFlags,
		composite_alpha: CompositeAlphaFlags,
		present_mode: PresentMode,
		old_swapchain: Option<&Self>,
	) -> (Arc<Self>, impl Iterator<Item = Arc<SwapchainImage>>) {
		let queue_family_indices: Vec<_> = queue_families
			.into_iter()
			.inspect(|qfam| assert!(device.physical_device() == qfam.physical_device()))
			.map(|qfam| qfam.idx)
			.collect();

		let image_sharing_mode =
			if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE };

		let ci = vk::SwapchainCreateInfoKHR::builder()
			.surface(surface.vk)
			.min_image_count(min_image_count)
			.image_format(image_format)
			.image_color_space(image_color_space)
			.image_extent(image_extent)
			.image_array_layers(1)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(image_sharing_mode)
			.queue_family_indices(&queue_family_indices)
			.pre_transform(pre_transform)
			.composite_alpha(composite_alpha)
			.present_mode(present_mode)
			.clipped(true)
			.old_swapchain(old_swapchain.map(|x| x.vk).unwrap_or(vk::SwapchainKHR::null()));
		let vk = unsafe { device.khr_swapchain.create_swapchain(&ci, None) }.unwrap();
		let images = unsafe { device.khr_swapchain.get_swapchain_images(vk) }.unwrap();

		let swapchain = unsafe { Swapchain::from_vk(device.clone(), surface, vk) };

		let swapchain2 = swapchain.clone();
		let images = images.into_iter().map(move |vk| Arc::new(SwapchainImage { swapchain: swapchain2.clone(), vk }));

		(swapchain, images)
	}

	pub fn acquire_next_image(self: &Arc<Self>, timeout: u64) -> Result<(u32, bool, AcquireFuture<T>), vk::Result> {
		let semaphore = Semaphore::new(self.device.clone());
		let (image_idx, suboptimal) =
			unsafe { self.device.khr_swapchain.acquire_next_image(self.vk, timeout, semaphore.vk, vk::Fence::null()) }?;
		Ok((image_idx, suboptimal, AcquireFuture { _swapchain: self.clone(), semaphore }))
	}

	pub fn present_after(
		semaphores: Vec<SemaphoreFuture>,
		queue: Arc<Queue>,
		swapchains: &[Arc<Swapchain<T>>],
		image_indices: &[u32],
	) -> Result<bool, vk::Result> {
		let semaphore_vks: Vec<_> = semaphores.iter().map(|x| x.semaphore().vk).collect();
		let swapchain_vks: Vec<_> = swapchains.iter().map(|x| x.vk).collect();

		// TODO: check individual results
		let ci = vk::PresentInfoKHR::builder()
			.wait_semaphores(&semaphore_vks)
			.swapchains(&swapchain_vks)
			.image_indices(image_indices);
		let suboptimal = unsafe { queue.device.khr_swapchain.queue_present(queue.vk, &ci) }?;

		Ok(suboptimal)
	}

	pub fn recreate<'a>(
		&self,
		min_image_count: u32,
		image_format: Format,
		image_color_space: ColorSpace,
		image_extent: Extent2D,
		queue_families: impl IntoIterator<Item = QueueFamily>,
		pre_transform: SurfaceTransformFlags,
		composite_alpha: CompositeAlphaFlags,
		present_mode: PresentMode,
	) -> (Arc<Swapchain<T>>, impl Iterator<Item = Arc<SwapchainImage>>) {
		let queue_family_indices: Vec<_> = queue_families
			.into_iter()
			.inspect(|qfam| assert!(self.device.physical_device() == qfam.physical_device()))
			.map(|qfam| qfam.idx)
			.collect();

		let image_sharing_mode =
			if queue_family_indices.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE };

		let ci = vk::SwapchainCreateInfoKHR::builder()
			.surface(self.surface.vk)
			.min_image_count(min_image_count)
			.image_format(image_format)
			.image_color_space(image_color_space)
			.image_extent(image_extent)
			.image_array_layers(1)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(image_sharing_mode)
			.queue_family_indices(&queue_family_indices)
			.pre_transform(pre_transform)
			.composite_alpha(composite_alpha)
			.present_mode(present_mode)
			.clipped(true)
			.old_swapchain(self.vk);
		let vk = unsafe { self.device.khr_swapchain.create_swapchain(&ci, None) }.unwrap();
		let images = unsafe { self.device.khr_swapchain.get_swapchain_images(vk) }.unwrap();

		let swapchain = unsafe { Swapchain::from_vk(self.device.clone(), self.surface.clone(), vk) };
		let swapchain2 = swapchain.clone();
		let images = images.into_iter().map(move |vk| Arc::new(SwapchainImage { swapchain: swapchain2.clone(), vk }));

		(swapchain, images)
	}

	pub fn surface(&self) -> &Arc<Surface<T>> {
		&self.surface
	}

	unsafe fn from_vk(device: Arc<Device>, surface: Arc<Surface<T>>, vk: vk::SwapchainKHR) -> Arc<Self> {
		Arc::new(Self { device, surface, vk })
	}
}
impl<T> Drop for Swapchain<T> {
	fn drop(&mut self) {
		unsafe { self.device.khr_swapchain.destroy_swapchain(self.vk, None) };
	}
}
impl<T> SwapchainAbstract for Swapchain<T> {
	fn device(&self) -> &Arc<Device> {
		&self.device
	}
}

pub struct SwapchainImage {
	swapchain: Arc<dyn SwapchainAbstract + Send + Sync>,
	vk: vk::Image,
}
impl ImageAbstract for SwapchainImage {
	fn device(&self) -> &Arc<Device> {
		&self.swapchain.device()
	}

	fn vk(&self) -> vk::Image {
		self.vk
	}
}

pub struct AcquireFuture<T> {
	_swapchain: Arc<Swapchain<T>>,
	semaphore: Arc<Semaphore>,
}
impl<T> GpuFuture for AcquireFuture<T> {
	fn device(&self) -> &Arc<Device> {
		&self.semaphore.device
	}

	fn flush(&mut self) {
		todo!()
	}

	fn build_submission(&mut self) -> SubmitState {
		let mut submit = SubmitState::new();
		submit.wait_semaphore(&self.semaphore, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
		submit
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		todo!()
	}
}

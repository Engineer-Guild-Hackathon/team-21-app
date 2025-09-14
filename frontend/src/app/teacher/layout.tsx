import TeacherNavbar from '../../components/TeacherNavbar';

export default function TeacherLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <TeacherNavbar />
      <main>{children}</main>
    </div>
  );
}
